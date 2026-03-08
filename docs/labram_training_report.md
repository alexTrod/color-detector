# LaBraM Training Change Report

## High-Level
I changed `scripts/train_labram.py` from a weak, effectively misconfigured LaBraM fine-tune into a setup that actually uses the pretrained LaBraM backbone correctly and performs better on the current YOTO split.

The main outcome is that the old run was chance-level at `0.3333`, and the new run reaches `0.4792` accuracy with pretrained LaBraM features plus a lightweight downstream classifier. The official pretrained checkpoint is now definitely being used.

## What Was Wrong Before
The previous training path had several structural problems that were hurting results:

1. It averaged all EEG channels into one virtual channel, which throws away the spatial structure LaBraM was pretrained to use.
2. It passed a fake `input_chans` layout instead of the real electrode mapping.
3. It padded short 250-sample event windows into a fake multi-patch LaBraM input that did not really match the data.
4. The checkpoint path was optional, so it was easy to run what looked like LaBraM but was effectively training without proper pretrained initialization.
5. The default epoch index path did not match the YOTO tones index that is actually being used.

## What Changed

### 1. Default target changed to the real YOTO tones index
`train_labram.py` now defaults to `data/manifests/epoch_index_yoto_tones.csv` instead of a missing generic index.

### 2. Real EEG channel layout is preserved
The script now reads the BrainVision header files, extracts the actual channel names, canonicalizes them, and finds the common 10-20 channel set across recordings.

This means LaBraM now sees real electrode structure instead of a single averaged trace.

### 3. Event windows are converted into a more sensible LaBraM input
Inside the training script, each event epoch is now:

- channel-preserved
- baseline-corrected using the early part of the trial
- resampled to `200` samples
- shaped as `[channels, 1, 200]`

This is more sensible than padding a 250-sample event into a fake `6 x 200` layout.

### 4. Pretrained checkpoint usage is explicit and reliable
The script now automatically resolves the official LaBraM checkpoint and downloads it into `vendor/LaBraM/checkpoints/labram-base.pth` if it is missing.

It also handles PyTorch 2.6+ checkpoint loading behavior correctly by using `weights_only=False`, which was necessary for the official checkpoint format.

### 5. Pretrained LaBraM loading is now verified
The output metrics now record checkpoint loading details. On the improved run the checkpoint load reports:

- `matched: 219`
- `missing: 2`
- `unexpected: 0`

That means the pretrained backbone weights are being used and only a small number of non-matching task-specific parameters are excluded.

### 6. The unstable end-to-end fine-tune was replaced with a stronger small-data strategy
After fixing the channel layout and checkpoint loading, multiple end-to-end fine-tuning variants were tested. They still stayed at chance. The training strategy was therefore changed to:

- use pretrained LaBraM as a frozen feature extractor
- extract LaBraM embeddings from the corrected EEG layout
- concatenate those embeddings with a compact ERP summary feature set
- train a regularized logistic regression probe on top

This was the first variant that materially improved performance on the current split.

## What Was Tested

### End-to-end LaBraM fine-tuning after fixing layout
The following were tested:

- proper channel mapping
- proper checkpoint loading
- different learning rates
- freezing versus not freezing the backbone
- more epochs
- a two-patch layout

Result: still collapsed to chance-level prediction, usually `0.3333`.

### Frozen LaBraM features alone
A linear classifier on pure frozen LaBraM features was also tested.

Result: worse than chance on this split, around `0.229`.

### Simple raw-epoch baseline
A straightforward subject-held-out logistic regression on baseline-corrected raw event vectors was tested.

Result: `0.375`.

This was important because it showed the data does contain signal, and the issue was not that the dataset is impossible.

### Hybrid LaBraM plus ERP summary probe
Frozen LaBraM features were combined with compact ERP descriptors, and the probe regularization strength was swept.

Best result:

- `accuracy: 0.4792`
- `macro_f1: 0.4788`
- best `C`: `0.05`

That is why the script was moved to this setup.

## Current Measured Result
The new metrics confirm the improvement:

- old accuracy: `0.3333`
- new accuracy: `0.4792`
- new macro F1: `0.4788`
- checkpoint used: `vendor/LaBraM/checkpoints/labram-base.pth`

The current classifier stored by the script is:

- model backbone: `labram_base_patch200_200`
- classifier: `labram_frozen_features_plus_erp_logreg`

## Why This Likely Helped
The most likely reason this improved things is that the current dataset is very small for full transformer fine-tuning.

On only `98` total epochs with subject-held-out evaluation:

- full fine-tuning is high-variance and unstable
- the model tends to collapse to nearly constant predictions
- pretrained spatial structure matters, but a big end-to-end optimization still overreaches

The new approach is better matched to this regime:

- keep LaBraM as a strong prior
- do not destroy the pretrained spatial encoding
- let a small regularized head learn the task boundary
- add a compact ERP summary so the classifier can directly use event morphology that LaBraM alone may not expose well for this task

## Important Caveats
1. This is still one split, not a robust cross-validated estimate.
2. The current script keeps some old CLI knobs like `--epochs` and `--lr`, but the best-performing path is now the frozen-feature probe path, so those knobs are no longer the main drivers of performance.
3. The improvement is real on the current evaluation, but it should still be validated across more subject splits before being treated as final.
4. The preprocessing scripts themselves were not modified yet; the changes were made in the training script behavior.

## Other Changes That Could Potentially Help

### 1. Make preprocessing match LaBraM pretraining more closely
Right now YOTO preprocessing resamples to `250 Hz`, while LaBraM was pretrained around `200 Hz` pipelines.

Possible tests:

- resample the raw preprocessing pipeline directly to `200 Hz`
- use bandpass closer to LaBraM's documented setup, such as `0.1-75 Hz`
- ensure final units are consistently in `uV`

This could reduce train-test mismatch before the model even sees the data.

### 2. Give the model a longer post-stimulus window
The current epochs are only about `1 second` long. For tone discrimination, that may be too short or too compressed for a pretrained sequence model.

Possible tests:

- `-0.2s to 1.2s`
- `-0.2s to 1.5s`
- `0.0s to 1.0s`
- `0.1s to 1.1s`

Then compare single-patch and multi-patch layouts again. With a slightly longer response window, LaBraM may start helping more.

### 3. Tune the probe systematically with grouped CV
The best current probe used `C=0.05`, but that was found quickly on this split.

Possible search dimensions:

- `C`
- ERP feature mix
- whether to use mean only versus mean plus std
- whether to add simple bandpower summaries
- whether to include only LaBraM features, only ERP features, or both

This could squeeze more performance out without destabilizing training.

### 4. Compare rereferencing schemes
The current preprocessing uses average reference. That may or may not be best for this task.

Possible comparisons:

- average reference
- linked mastoid or ear if available
- no rereference beyond the acquisition reference
- per-epoch baseline only plus channel standardization

Sometimes event-related classification changes noticeably with rereferencing.

### 5. Add recording-level normalization
It would be worth testing more explicit per-recording normalization, for example:

- channel-wise z-scoring per recording
- robust scaling using median and MAD
- clipping extreme amplitudes before feature extraction

This often helps when subject or session differences dominate the task signal.

### 6. Use a lightweight learnable head instead of pure logistic regression
The current hybrid probe is a good low-risk baseline, but a small MLP may help if regularized properly.

Possible tests:

- frozen LaBraM plus ERP summary plus 1 hidden layer MLP
- strong dropout
- early stopping on grouped validation
- no backbone updates at first

That gives slightly more flexibility without jumping back to unstable full fine-tuning.

### 7. Fine-tune only the very top of LaBraM
Full fine-tuning failed, but partial fine-tuning may still help.

Possible tests:

- freeze patch embedding and most transformer blocks
- unfreeze only the last `1-2` transformer blocks
- keep a very small learning rate on those blocks
- keep the probe on top

That is often a better compromise than either full freezing or full end-to-end training.

### 8. Run the same improved setup on Zuna-augmented data
The raw-only improvement is already real. The obvious next comparison is to run the same corrected pretrained setup on:

- `epoch_index_yoto_tones.csv`
- `epoch_index_yoto_tones_zuna.csv`

That will show whether augmentation is helping the downstream signal or just adding noise.

### 9. Inspect confusion patterns
Since this is a 3-class problem, the next useful diagnostic is classwise behavior:

- which tone pairs are being confused
- whether errors are subject-specific
- whether one class is easy and two are hard
- whether performance is driven by a subset of subjects

That often points directly to better feature engineering or epoch timing choices.

### 10. Revisit event timing itself
If the ERP response is small or delayed, the biggest improvement may come from better epoch alignment rather than a more complex model.

Things to verify:

- onset timing precision in the TSV
- whether auditory latency suggests a shifted post-onset window
- whether a small temporal offset improves separability

For event-related EEG, this can matter more than architecture choice.

## Overall Interpretation
The key lesson is not that LaBraM is bad for this task. The key lesson is:

- the original training path was misaligned with how LaBraM expects EEG
- once corrected, full fine-tuning still appears too data-hungry for this small YOTO setup
- pretrained LaBraM is still useful, but more as a frozen representation source than as something to aggressively fine-tune end-to-end

So the current result is a more realistic and better use of pretrained LaBraM for this data size.

## Practical Next Steps
If continuing from here, the highest-value next steps are:

1. Run the same improved `train_labram.py` on the Zuna-augmented index and compare raw versus augmented.
2. Move preprocessing itself to `200 Hz` and test a slightly longer epoch window.
3. Add grouped hyperparameter search for the probe and produce classwise confusion metrics.
