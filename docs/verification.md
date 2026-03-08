# EEG Pipeline Verification Checklist

## Phase 1: Preprocessing Validation

### 1. Data Shape & Continuity
- [ ] Load preprocessed data: shape should be `(n_windows, n_channels, n_samples)`
- [ ] Verify `n_samples = 256 Hz × 5s = 1280`
- [ ] All 5 subjects have same channel count
- [ ] Load YOTO derivatives and compare channel ordering

### 2. Amplitude Characteristics
- [ ] Mean per channel ≈ 0 (after re-referencing)
- [ ] Std per channel: 10-50 µV (typical range)
- [ ] No dead channels (std < 0.1 µV)
- [ ] No excessively noisy channels (std > 3× global std)

### 3. Frequency Content ⚠️ CRITICAL
- [ ] Compute PSD on sample windows
- [ ] **90%+ power in 0.5-45 Hz band**
- [ ] No visible 50/60 Hz line noise
- [ ] Spectral peaks match YOTO derivatives
- [ ] Raw signal visually looks like clean EEG

### 4. Window Continuity
- [ ] No sudden jumps at window boundaries
- [ ] Discontinuity < 2× within-window std
- [ ] Smooth transitions between consecutive windows

### 5. Outlier Detection
- [ ] Compute std for each window
- [ ] Identify windows where std > mean + 4σ
- [ ] Flag these as potential artifacts
- [ ] Expected: <5% outlier windows

### 6. Comparison to YOTO Derivatives
- [ ] Amplitude ranges similar
- [ ] Channel count matches
- [ ] Channel ordering matches 10-20 system
- [ ] Frequency content aligns

---

## Phase 2: Augmentation Validation

### 1. Data Integrity
- [ ] No NaN values: `np.isnan(augmented_data).any() == False`
- [ ] No Inf values: `np.isinf(augmented_data).any() == False`
- [ ] Amplitude in reasonable range (-200 to 200 µV)

### 2. Statistical Preservation
- [ ] Mean shift: `|mean(original) - mean(augmented)| < 5 µV`
- [ ] Std change: `|std(original) - std(augmented)| < 10 µV`
- [ ] Correlation: `corr(original, augmented) > 0.7`
- **→ If any fail: augmentation is too aggressive**

### 3. Frequency Domain
- [ ] Compute PSD on augmented data
- [ ] Spectral peaks match original data
- [ ] Power spectrum visually similar
- [ ] No unexpected frequency changes

### 4. Distribution Similarity
- [ ] Plot histogram: original vs augmented
- [ ] Distributions look similar (same shape)
- [ ] No bimodal or unusual patterns

### 5. Data Leakage Detection ⚠️ CRITICAL
- [ ] Total augmented samples: `augmented.shape[0]`
- [ ] Training samples (pre-aug): `len(train_idx)`
- [ ] Expected augmented size: `len(train_idx) × aug_factor`
- **→ If actual >> expected: augmentation leaked to val/test (BAD!)**

---

## Phase 3: Data Preparation

### 1. Labels
- [ ] Labels exist for all windows
- [ ] Manually spot-check 10 windows against raw data
- [ ] Labels match task (color vs pitch classification)

### 2. Train/Val/Test Split
- [ ] Split is stratified (class balance preserved in each set)
- [ ] No overlap between splits
- [ ] ~70% train, ~15% val, ~15% test (or your split)
- [ ] **Augmentation only applied to training set**

### 3. Normalization
- [ ] **Per-subject normalization applied** (critical for EEG!)
- [ ] Not per-sample or per-window
- [ ] Z-score normalization: `(x - mean) / std`

---

## Phase 4: Model Training Checks

### 1. Before Training
- [ ] Labels verified (spot-check completed)
- [ ] Split is stratified
- [ ] Per-subject normalization applied
- [ ] Augmentation in train set only

### 2. During Training
- [ ] Training loss decreases (not stuck or NaN)
- [ ] Validation loss follows training loss
- [ ] No sudden spikes in loss
- [ ] Expected: val loss ≈ train loss (with 5 subjects, some overfitting OK)

### 3. After Training: Accuracy Check
- [ ] **Model achieves 60-85% test accuracy** (expected range for 5 subjects)
- [ ] If < 50%: check labels (likely wrong)
- [ ] If > 95%: check for label leakage
- [ ] Both models should be in reasonable range

### 4. Generalization Analysis
- [ ] Train vs Val gap: `val_acc - train_acc > -0.05` (no severe overfitting)
- [ ] Val vs Test gap: small (good generalization)
- [ ] If test << val: overfitting occurred

### 5. Per-Class Performance
- [ ] Compute precision, recall, F1 per class
- [ ] Identify which classes are harder (low F1)
- [ ] Check confusion matrix for systematic errors
- [ ] Some class variation is expected

---

## Phase 5: Original vs Augmented Comparison

### 1. Performance Difference
- [ ] Augmentation improvement: `(acc_aug - acc_orig) / acc_orig × 100`
- [ ] Expected: +2-10% boost
- [ ] If negative: check for leakage or too-aggressive augmentation

### 2. Per-Class Analysis
- [ ] Which classes benefit most from augmentation?
- [ ] Are any classes worse with augmentation?
- [ ] Compare precision/recall/F1 per class

### 3. Learning Stability
- [ ] Which model has smoother training curve?
- [ ] Which converges faster?
- [ ] Any divergence or instability?

---

## Critical Red Flags 🚨

| Flag | Diagnosis | Action |
|------|-----------|--------|
| Dead channels (std < 0.1 µV) | Hardware/preprocessing error | Check electrode impedance |
| Power outside 0.5-45 Hz | Filter not applied or wrong order | Verify preprocessing pipeline |
| Mean amplitude >> 0 | Re-referencing failed | Check re-reference code |
| NaN in augmented data | Zuna bug | Debug augmentation code |
| Augmented accuracy < original | Data leakage or too aggressive | Verify no aug in val/test |
| Test accuracy > 95% | Label leakage | Check split integrity |
| Test accuracy < 50% | Wrong labels or impossible task | Manually verify 10 labels |
| Discontinuity > 2σ | Windowing error | Check window boundaries |

---

## Final Checklist Before Model Training

**Preprocessing:**
- [ ] All 5 subjects: correct shape
- [ ] All 5 subjects: amplitude 10-50 µV
- [ ] All 5 subjects: no dead channels
- [ ] All 5 subjects: >90% power in 0.5-45 Hz
- [ ] Matches YOTO derivatives

**Augmentation:**
- [ ] No NaN/Inf values
- [ ] Statistics preserved (mean, std, correlation)
- [ ] Only in training set (no leakage)

**Data Preparation:**
- [ ] Labels verified (spot-check 10)
- [ ] Split stratified and no overlap
- [ ] Per-subject normalization applied

**→ If all checks pass: Ready for training**
**→ If any fail: Debug that phase first**

---

## Typical Expected Results (5 subjects)

| Metric | Expected Range |
|--------|-----------------|
| Test Accuracy | 60-85% |
| Train vs Val gap | < 5% |
| Outlier windows | < 5% |
| Power in 0.5-45 Hz | > 90% |
| Augmentation boost | +2-10% |
| Power spectrum similarity | > 0.7 |

---

## What to Compare Against

**YOTO Derivatives (if available):**
- Amplitude ranges
- Channel count & ordering
- Frequency content
- Preprocessing approach

**Your Two Models:**
- Original preprocessing → Model 1
- Same preprocessing + Zuna augmentation → Model 2
- Compare test accuracy, per-class metrics, convergence speed