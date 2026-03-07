# Human Activity State Classification

## Overview

This project implements a Hidden Markov Model (HMM) to automatically classify and predict human activity states from sensor data collected via smartphone accelerometer and gyroscope readings. The system can recognize and distinguish between different physical activities including standing, walking, jumping, and remaining still.

## Project Context

Activity recognition from wearable sensors is a fundamental capability with applications in health monitoring, fitness tracking, and smart home systems. This project addresses the challenge of building robust models that can infer real-world human activity states from continuous sensor streams, where the underlying activity may not be directly observable.

## Project Structure

```
Human-Activities-State/
├── README.md                                 # Project documentation
├── LICENSE                                   # MIT License
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Git ignore rules
├── Team_Task_Sheet_HMM_Group28.pdf           # Project task specifications
├── notebooks/
│   └── activity_recognition_hmm.ipynb        # Main analysis notebook
├── data/
│   ├── train/                                # Training data (62 files)
│   │   ├── jumping/                          # 15 jumping recordings
│   │   ├── standing_waist/                   # 17 standing recordings
│   │   ├── still/                            # 15 still recordings
│   │   └── walking/                          # 15 walking recordings
│   └── test/                                 # Test data (5 files)
│       ├── setB_Jumping_*.csv                # 3 jumping test recordings
│       └── setB_Standing_*.csv               # 2 standing test recordings
└── results/                                  # Generated visualizations
    ├── confusion_matrix.png
    ├── per_activity_metrics.png
    ├── hmm_transition_matrices.png
    ├── sensitivity_specificity.png
    └── decoded_activity_sequences.png
```

## Dataset Description

The dataset is split into **training** and **test** sets organized by activity class:

**Training Data** (`data/train/`):
- **Jumping** (15 sequences): 5 original + 10 additional recordings from 2026-03-04 and 2026-03-07
- **Standing** (17 sequences): 7 original + 10 additional recordings from 2026-03-04 and 2026-03-07
- **Still** (15 sequences): 5 original + 10 additional recordings from 2026-03-04 and 2026-03-07
- **Walking** (15 sequences): 5 original + 10 additional recordings from 2026-03-04 and 2026-03-07

**Test Data** (`data/test/`):
- **Jumping**: 3 files (setB recordings from 2025-10-24)
- **Standing**: 2 files (setB recordings from 2025-10-26)
- Files from test activities in a flat folder structure
- Used for unbiased model evaluation without temporal data leakage

Each CSV file contains:
- Accelerometer readings: `accel_x`, `accel_y`, `accel_z` (m/s²)
- Gyroscope readings: `gyro_x`, `gyro_y`, `gyro_z` (rad/s)
- Timestamp information
- Sampling rate metadata

**Total Dataset:** 58 training sequences + 5 test sequences across 4 activities with 1,667 total windows of sensor data

## Methodology

### 1. Feature Extraction
Extract comprehensive time-domain and frequency-domain features from sliding windows (50 samples with 50% overlap):

**Time-Domain Features (per axis):**
- Mean, standard deviation, variance
- Min, max, range (max - min)
- Root Mean Square (RMS)
- Mean Absolute Deviation (MAD)
- Skewness and kurtosis
- Interquartile Range (IQR)
- Signal energy (sum of squared values)

**Frequency-Domain Features (per axis):**
- Spectral energy (sum of FFT magnitudes squared)
- Spectral entropy (normalized entropy of frequency distribution)
- Dominant frequency
- Frequency centroid
- Energy in activity-relevant bands (0-2 Hz, 2-5 Hz, 5-10 Hz)

**Cross-Axis Features:**
- Signal Magnitude Area (SMA) for accelerometer and gyroscope signals
- Correlation coefficients between axes (X-Y, X-Z, Y-Z)

**Total Feature Dimension:** 119 features per window

### 2. Model Architecture

We implement a **one-HMM-per-class** architecture with 4 separate Gaussian HMMs:

| Component | Specification |
|-----------|---------------|
| **Number of HMMs** | 4 (one per activity: Standing, Walking, Jumping, Still) |
| **Hidden States per HMM** | 3 internal states (Z ∈ {0, 1, 2}) |
| **Observations** | 119-dimensional feature vectors per window |
| **Emission Model** | Diagonal-covariance Gaussian N(μ_s, Σ_s) |
| **Transition Probability Matrix (A)** | 3×3 matrix per activity; A[i,j] = P(z_{t+1}=j \| z_t=i) |
| **Emission Means (μ)** | 3×119 matrix; μ_s = mean of observations in state s |
| **Emission Covariances (Σ)** | 3×119 diagonal matrix; Σ_s = variance per feature in state s |
| **Initial State Probability (π)** | 3-element vector; π_s = P(z_0=s) |

**Key Design Choice:** Each activity has its own HMM with independent parameters. This allows the model to learn activity-specific temporal dynamics. The 3 internal hidden states capture within-activity transitions (e.g., different phases of jumping).

### 3. Model Training

**Algorithm:** Baum-Welch (Expectation-Maximization)

**E-step:** Compute expected sufficient statistics via forward-backward algorithm (log-space for numerical stability)
- Forward pass: compute forward probabilities using Viterbi-style log-sum-exp
- Backward pass: compute backward probabilities
- Compute posteriors γ(t) = P(z_t | X) and pairwise posteriors ξ(t,i,j) = P(z_t=i, z_{t+1}=j | X)

**M-step:** Update parameters to maximize expected log-likelihood
- Update means and variances using weighted statistics
- Update transition matrix from expected state transitions
- Update initial probabilities from expected initial state distribution

**Convergence:** Training stops when log-likelihood improvement < 10⁻⁵ (typically 3-6 iterations)

**Warm-Start:** Initialize parameters using labeled training windows to give Baum-Welch a good starting point

### 4. Inference (Viterbi Decoding)

**Class-level decoding:**
For each test sequence X = (x_1, x_2, ..., x_T):

1. For each of the 4 class HMMs: 
   - Run Viterbi algorithm to find the most likely internal state sequence
   - Compute the log-probability log P(X | HMM_c) of the observations under that class model
2. Select the class c* with the highest log-probability: c* = argmax_c log P(X | HMM_c)
3. Assign all windows in the sequence to class c*

**Why this approach:** By operating on complete sequences, the transition matrix A influences the decoding. If a state sequence is unlikely under a class HMM (based on learned transitions), that class gets a lower score, reducing false positives.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Mugisha-isaac/Human-Activities-State.git
cd Human-Activities-State
```

2. **Create and activate a virtual environment:**
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook notebooks/activity_recognition_hmm.ipynb
```

The notebook executes the following steps automatically:

1. **Environment Setup** - Import NumPy, pandas, scikit-learn, matplotlib, seaborn
2. **Data Loading** - Read training files from `data/train/` organized by activity
3. **Test Data Loading** - Read test files from `data/test/` for unbiased evaluation
4. **Feature Extraction** - Compute 119 features per window from accelerometer and gyroscope signals
5. **Data Preprocessing** - Standardize features using StandardScaler fitted on training data only
6. **HMM Training** - Train 4 separate Gaussian HMMs using Baum-Welch algorithm
7. **Sequence-Level Decoding** - Run Viterbi algorithm on test sequences to assign activity labels
8. **Evaluation** - Compute accuracy, precision, recall, F1-score, confusion matrix, sensitivity, specificity
9. **Visualization** - Generate and save 5 professional plots to `results/`

## Key Deliverables

- `activity_recognition_hmm.ipynb` - Complete Jupyter notebook with data loading, feature extraction, HMM training, and comprehensive evaluation
- `results/` folder containing:
  - `confusion_matrix.png` - Classification performance visualization
  - `per_activity_metrics.png` - Precision, recall, and F1-score charts
  - `hmm_transition_matrices.png` - Learned state transition probabilities
  - `sensitivity_specificity.png` - Sensitivity and specificity analysis
  - `decoded_activity_sequences.png` - Actual vs. predicted activity sequences

## Evaluation Metrics

Model performance is evaluated using comprehensive metrics:
- **Overall Accuracy:** Percentage of correctly classified windows across all activities
- **Per-Activity Metrics:** Precision, recall, and F1-score for each activity class
- **Confusion Matrix:** Visual representation of correct and misclassified predictions for each activity
- **Sensitivity & Specificity:** True positive rate and true negative rate for evaluating detection capability and false alarm rates
- **Transition Probability Analysis:** Examination of learned HMM state transition patterns to validate temporal dynamics

## Model Results

### Performance Summary

Classification accuracy achieved on test data:
- **Overall Accuracy:** 100.00% (248/248 windows correctly classified)
- **Macro Precision:** 0.5000 (2/4 activities present in test set)
- **Macro Recall:** 0.5000 (2/4 activities present in test set)
- **Macro F1-Score:** 0.5000 (2/4 activities present in test set)
- **Average Sensitivity (Recall):** 0.5000 (perfect detection on tested activities)
- **Average Specificity:** 1.0000 (zero false positives)

### Dataset Statistics
- **Total Windows:** 1,667
- **Training Sequences:** 58 files
- **Training Windows:** 1,419 (balanced across 4 activities)
- **Test Sequences:** 5 files
- **Test Windows:** 248 (Jumping: 133, Standing: 115)
- **Feature Dimension:** 119

### Per-Activity Performance

| Activity  | Precision | Recall | F1-Score | Sensitivity | Specificity | Test Samples | TP  | FP  | TN  | FN  |
|-----------|-----------|--------|----------|-------------|-------------|--------------|-----|-----|-----|-----|
| Jumping   | 1.0000    | 1.0000 | 1.0000   | 1.0000      | 1.0000      | 133          | 133 | 0   | 115 | 0   |
| Standing  | —         | —      | —        | 1.0000      | 1.0000      | 115          | 115 | 0   | 133 | 0   |
| Walking   | —         | —      | —        | 0.0000      | 1.0000      | 0            | 0   | 0   | 248 | 0   |
| Still     | —         | —      | —        | 0.0000      | 1.0000      | 0            | 0   | 0   | 248 | 0   |

**Note:** Test dataset contained only Jumping and Standing activities. Perfect sensitivity (1.0000) and specificity (1.0000) indicate flawless classification of activities present in test data.

### Key Visualizations

The analysis includes comprehensive visualizations saved in the `results/` folder:

#### 1. **Confusion Matrix** (`confusion_matrix.png`)
Shows the classification performance for each activity on test data. The model achieved:
- **Jumping:** All 133 test windows correctly classified as Jumping (100% accuracy)
- **Standing:** All 115 test windows correctly classified as Standing (100% accuracy)
- **Zero misclassifications** across all predicted windows
- Overall perfect discrimination between activities with no off-diagonal errors

#### 2. **Per-Activity Metrics** (`per_activity_metrics.png`)
Displays precision, recall, and F1-score for each activity:
- **Jumping:** Perfect scores (Precision=1.0, Recall=1.0, F1=1.0)
- **Standing:** Perfect scores (Precision=1.0, Recall=1.0, F1=1.0)
- **Walking & Still:** Not present in test set (no predictions made)
- Demonstrates perfect discriminative capability on tested activities

#### 3. **HMM Transition Matrices** (`hmm_transition_matrices.png`)
Visualizes the learned state transition probabilities for each activity class's 3 internal hidden states. Shows distinct temporal patterns:
- **Jumping:** Balanced state transitions with cycle through all three states (s0 → s1 → s2 pattern)
- **Standing:** Higher self-transition probability on s1 (0.576), indicating sustained posture with minimal state changes
- **Walking:** Strong self-loop on state s1 (0.948), reflecting continuous cyclic walking motion
- **Still:** Perfect deterministic transitions (s0→s1→s2→s2), indicating locked stable posture
- Each activity learns characteristic temporal dynamics reflecting its inherent motion pattern

#### 4. **Sensitivity and Specificity** (`sensitivity_specificity.png`)
Compares the true positive rate (sensitivity) and true negative rate (specificity) for each activity:
- **Jumping:** Sensitivity = 1.0000, Specificity = 1.0000 (perfect detection with zero false positives)
- **Standing:** Sensitivity = 1.0000, Specificity = 1.0000 (perfect detection with zero false positives)
- **Walking & Still:** Not evaluated (absent from test set)
- Average Specificity = 1.0000 demonstrates zero false alarms across all activities

#### 5. **Decoded Activity Sequences** (`decoded_activity_sequences.png`)
Shows example test recordings with true labels (solid lines) versus predicted labels (dashed lines):
- 4 example sequences displayed (2 Standing and 2 Jumping from test set)
- **Standing sequences:** Perfectly predicted throughout entire sequence
- **Jumping sequences:** Perfectly predicted with consistent activity label across all windows
- Perfect alignment between true and predicted labels confirms accurate sequence-level inference

## Key Improvements Over Baseline

This implementation addresses critical design issues from naive sliding-window classifiers:

1. **Full-Sequence Viterbi:** A naive window-by-window approach would call Viterbi independently on each window with only 10-sample context, destroying temporal coherence and making classes indistinguishable. This implementation:
   - Decodes **entire sequences as atomic units**
   - Allows transition probabilities A to enforce consistent state sequences
   - Scores sequences **holistically** using class log-likelihoods

2. **Baum-Welch EM vs. Static Estimation:** Instead of manually counting label transitions, the model automatically discovers optimal transition and emission parameters via:
   - E-step: Compute expected state posteriors using forward-backward algorithm
   - M-step: Update parameters to maximize data likelihood
   - Result: proper statistical inference under the HMM generative model

3. **Sequence-Level Train/Test Split:**
   - Training and test data split at file (sequence) level, not window level
   - Prevents temporal leakage: windows from same sequence can't appear in both train and test
   - Enables honest evaluation of sequence-level prediction capability

4. **One HMM Per Activity Class:** Rather than a single 4-state HMM with states = {Standing, Walking, Jumping, Still}:
   - Train **4 separate HMMs**, each with 3 internal states
   - Each HMM models the temporal micro-structure of one activity
   - At inference: class assignment determined by which HMM fits best
   - Decouples between-activity discrimination from within-activity dynamics

5. **Extended Feature Set (119 features):**
   - Time-domain: 13 features per axis (mean, std, skew, kurt, IQR, energy, etc.)
   - Frequency-domain: 7 features per axis (spectral energy, entropy, dominant freq, band energies)
   - Cross-axis: SMA and correlations between axes
   - Captures both static postures and dynamic motion patterns

## Analysis Points

- **Model Performance:** The classifier achieved **100% accuracy** (248/248 windows) on the test set, perfectly distinguishing between the tested activities (Jumping and Standing). This perfect classification demonstrates that the feature extraction and HMM architecture effectively capture activity-specific patterns.

- **Test Data Composition:** Test data included 248 total windows from two activities:
  - Jumping: 133 windows (3 test sequences from 2025-10-24)
  - Standing: 115 windows (2 test sequences from 2025-10-26)
  - No test samples for Walking or Still activities

- **Expanded Training Dataset:** Training data increased from original 23 files to **58 sequences** containing:
  - Training accuracy enabled by learning on 1,419 windows spread across all 4 activities
  - Balanced representation: Standing (30.2%), Walking (22.6%), Jumping (24.7%), Still (22.6%)
  - Extended temporal coverage with newly added files from 2026-03-07, improving generalization

- **Learned Transition Patterns:** HMM transition matrices reveal activity-specific temporal structure:
  - **Standing:** Higher probability of remaining in state s1 (0.576), indicating postural stability with occasional small movements
  - **Jumping:** More balanced transitions with progression through all states, reflecting repetitive jumping phases
  - **Walking:** Strong self-loop on s1 (0.948), capturing continuous rhythm of gait
  - **Still:** Deterministic transitions enforcing a fixed state sequence, representing stationary lock-in

- **Classification Stability:** Perfect specificity (1.0000) across all tested classes indicates zero false positives. When the model predicts an activity, it is always correct. This stability enables use in safety-critical applications.

- **Feature Effectiveness:** The **119-dimensional feature set** successfully captures:
  - Spectral characteristics distinguishing repetitive (jumping/walking) from static (standing/still) activities
  - Cross-axis correlations that differentiate body-aligned (standing) from unstructured (still) signals
  - Energy distribution patterns unique to each activity class

- **Sequence-Level Inference:** Using Viterbi decoding on complete sequences (rather than window-by-window classification) ensures that:
  - Temporal continuity constraints are respected (transition probabilities have effect)
  - Brief noisy windows are corrected by sequence context
  - Activity assignments are consistent across multi-window sequences

## Future Improvements

Potential architectural and algorithmic enhancements to increase robustness and generalization:

1. **Expand Test Data Coverage:** 
   - Current evaluation is limited to Jumping and Standing test activities
   - Collect/use additional test samples for Walking and Still to validate complete 4-class discrimination
   - Cross-participant testing: evaluate on data from different users to assess generalization across individuals

2. **Variable Number of States:** 
   - Current implementation uses fixed 3 hidden states per activity
   - Optimize state count per activity: Jumping might benefit from 4-5 states (multiple cycle phases), while Standing might need only 1-2 (minimal dynamics)
   - Use model selection criteria (BIC/AIC) to data-drive state count selection

3. **Mixture-of-Gaussians Emissions:** 
   - Replace diagonal Gaussian emissions with mixture models for complex multi-modal feature distributions
   - Better captures states with heterogeneous movement patterns within a single activity class

4. **Adaptive Window Size:** 
   - Currently fixed 50-sample windows with 50% overlap
   - Optimize for each activity: dynamic activities (jumping, walking) might need shorter windows (0.3-0.5s) for responsiveness, while static activities (standing, still) might benefit from longer windows (0.7-1.0s) for stability
   - Multi-scale features: extract features at multiple window scales and fuse decisions

5. **Larger and More Diverse Datasets:**
   - Collect more recordings per person and multiple diverse participants
   - Current: 58 training sequences from limited individuals
   - Target: 100+ sequences across 10+ participants with varying demographics, heights, weights
   - Cross-person evaluation to identify truly universal activity signatures

6. **Sequence-Level Features (Higher-Order Patterns):** 
   - Add features computed over multiple windows (e.g., trends, variance of window-means)
   - Capture longer-term dynamics beyond the 0.5-second window scale
   - Enable detection of transitions between activities (e.g., standing → walking start)

7. **Ensemble Models:** 
   - Train multiple HMM sets with different random seeds and model configurations
   - Average predictions across ensemble for improved robustness to sensor noise and individual user variation
   - Uncertainty estimates via ensemble disagreement

8. **Advanced Signal Processing:**
   - **Wavelet Features:** Discrete Wavelet Transform (DWT) to capture transient events (impacts during jumping)
   - **Time-Frequency Maps:** Spectrograms for activities with non-stationary frequency content
   - **Temporal Derivatives:** Feature velocity/acceleration for detecting rapid motion changes

9. **Semi-Supervised Learning:** 
   - Incorporate unlabeled sensor data to improve parameter estimates via EM with latent class labels
   - Leverage large amounts of low-cost unlabeled data to reduce dependence on hand-labeled sequences

10. **Real-Time Deployment Optimization:**
    - Implement sliding-window inference for online activity recognition (streaming data)
    - Quantization/pruning of 119 features to identify minimal sufficient feature subset
    - Target: ultra-low-latency mobile/edge inference with <10ms latency per window

11. **Transition Detection:**
    - Develop post-processing to detect activity transitions (e.g., Standing→Walking)
    - Currently: each sequence assigned single activity label
    - Future: temporal segmentation to identify exact transition boundaries

12. **Domain Adaptation:**
    - Train on one device/sensor configuration, adapt to new devices with minimal labeled data
    - Device-agnostic features (normalization, scaling) to handle sensor heterogeneity

## Project Deliverables

The final submission includes:

1. **Complete Jupyter Notebook** (`activity_recognition_hmm.ipynb`) featuring:
   - Hierarchical data loading from organized train/test folders preserving sequence boundaries
   - Comprehensive feature extraction (119 features per 50-sample window)
   - Custom Gaussian HMM implementation with Baum-Welch EM training
   - Sequence-level Viterbi decoding for whole-sequence classification
   - Full evaluation pipeline with 10+ quantitative metrics per activity
   - Professional visualizations saved to `results/`:
     - Confusion matrix for all 4 activities (100% accuracy on tested subset)
     - Per-activity precision, recall, F1-score, sensitivity, specificity
     - HMM transition matrices showing learned temporal patterns for each activity (3 hidden states)
     - Sensitivity and specificity analysis (perfect 1.0000 specificity)
     - Decoded activity sequences on real test recordings with true vs. predicted labels

2. **Organized Dataset** with expanded training data:
   - `data/train/` - 58 labeled sensor sequences organized by activity
     - Jumping: 15 sequences, 351 total windows
     - Standing: 17 sequences, 428 total windows
     - Still: 13 sequences, 320 total windows
     - Walking: 13 sequences, 320 total windows
   - `data/test/` - 5 test sequences (248 windows) from 2 activities (Jumping, Standing) for unbiased evaluation

3. **Result Visualizations** in `results/` folder:
   - `confusion_matrix.png` - Classification performance heatmap (248/248 correct)
   - `per_activity_metrics.png` - Precision/recall/F1 charts for all 4 activities
   - `hmm_transition_matrices.png` - Learned 3×3 temporal patterns per activity showing state dynamics
   - `sensitivity_specificity.png` - Detection capability analysis (mean sensitivity=0.5, specificity=1.0)
   - `decoded_activity_sequences.png` - 4 example test recordings showing perfect prediction alignment

4. **Updated README Documentation** with:
   - Complete expanded methodology explanation (119 features, 3-state HMMs)
   - Model architecture rationale and design choices
   - Updated performance results with 100% test accuracy
   - Detailed analysis of learned temporal patterns
   - 12 concrete future improvement suggestions
   - References to key design improvements over naive baselines

## Contributors

| Name | Role | Contributions |
|------|------|---------------|
| **Mugisha Isaac** | Lead Developer | HMM implementation, Baum-Welch/Viterbi algorithms, model training pipeline, initial data collection |
| **Michael Musembi** | Data & Evaluation | Additional sensor data collection (40 recordings), evaluation framework, documentation, repo maintenance |

## References

- Additional documentation on Hidden Markov Models
- Sensor data processing best practices
- Activity recognition benchmarks

---

*Last Updated: March 2026*
