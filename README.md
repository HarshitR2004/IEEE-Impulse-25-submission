# Seizure-type classification from short EEG segments

48-hour final-round project for IEEE Impulse 2025 (17–19 January). The task was four-class classification of 19-channel EEG, plus explainability, denoising, and synthetic data. This repository is the hackathon submission as it was run, together with a later audit of that code. The dataset is no longer available, so nothing here has been retrained. Every metric below is a number printed by an executed notebook.

I am looking for a research internship in EEG and machine learning. The part I would continue in a lab is the feature representation and a proper evaluation protocol, not the hackathon accuracy as a clinical claim.

## Problem

Each recording is 19 channels by 500 samples. Classes follow the organizer mapping:

| Label | Class |
|------:|-------|
| 0 | Normal |
| 1 | Complex partial seizures |
| 2 | Electrographic seizures |
| 3 | Video-detected seizures with no visual change on the EEG |

Split sizes in the notebooks: 5,608 train, 1,403 validation, 779 test. Training counts were 2,783 / 2,196 / 545 / 84. Validation was not mixed into the training loss. The test file `test_outputs.csv` contains hard labels only (779 files), so a test-set ROC-AUC cannot be recomputed from this repo.

## What is worth reading

The useful technical work is the 950-dimensional representation and the model trained on it.

Per recording the feature vector is:

- 209 time-domain features (11 statistics on each of 19 channels: mean, standard deviation, max, min, zero crossings, RMS, mean absolute value, kurtosis, skewness, Hjorth mobility, and a complexity statistic)
- 114 spectral features (Welch band power in delta, theta, alpha, beta, and gamma, plus 95% spectral edge, at an assumed sampling rate of 256 Hz)
- 285 wavelet features (db4, 4 levels, mean / standard deviation / energy of each coefficient block)
- 342 connectivity features (Pearson correlation and an unnormalized cross-correlation peak for every channel pair)

The classifier is a residual MLP with sigmoid feature gating, width 512. `Task 7/building-the-best-model.ipynb` reports 1,445,764 trainable parameters and 0 non-trainable parameters, which matches the module. On the validation set that notebook reports:

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| Normal | 0.98 | 0.99 | 0.98 | 696 |
| Complex partial | 0.99 | 0.97 | 0.98 | 549 |
| Electrographic | 0.99 | 0.99 | 0.99 | 137 |
| Video-detected | 1.00 | 1.00 | 1.00 | 21 |

Accuracy 0.98, balanced accuracy 0.9876, one-vs-rest ROC-AUC 0.9983. The same figures are reproduced in Task 8 from the saved checkpoint. Class 3 has 21 validation recordings, so that perfect cell is fragile.

The baseline in Task 6 is the contrast that makes this result interesting, and it is also the clearest bug in the repo. See the audit.

## Repository map

| Path | What was run |
|------|----------------|
| `Task 4/` | One recording per class, 19 channel plots plus an overlay, and the six time-domain statistics |
| `Task 5/` | FFT, 4-level db4 decomposition, spectrograms |
| `Task 6/` | Linear SVM on a Fourier summary and zero-crossing rate |
| `Task 7/` | 950-feature network, validation report, parameter count, `test_outputs.csv` |
| `Task 8/` | Attention-based channel ranking and a masking check |
| `Task 9/` | Wavelet denoising of the noisy training set, then the same classifier |
| `Task 10/` | Conditional convolutional GAN and a classifier trained on its samples |
| `Utility Scripts/` | Data loader, feature extractor, and `EEGClassifier` |

Notebooks were executed on Kaggle. Paths such as `/kaggle/input/...` will not resolve locally, and the `.npy` recordings are not in this repository.

## Later audit

After the event I reviewed the notebooks against the problem statement. The full notes are in [review/post-hackathon-audit.md](review/post-hackathon-audit.md). The points that change how a reader should use this repo:

1. **Task 6 Fourier axis.** Recordings are stored as `(samples, 19 channels, 500 time points)`. The FFT is taken on axis 1, across channels, and then averaged to a single number. Zero-crossing rate is averaged to a second number. The linear SVM therefore sees two scalars. Its validation numbers (accuracy 0.67, ROC-AUC 0.770, balanced accuracy 0.490, class 2 recall 0) are an honest score of that feature, not of a real spectrum.
2. **Task 8 channel ranking does not identify electrodes.** Gating is applied after a linear layer has mixed all 950 features into 512 units. Those units are then sliced into 19 chunks. Masking assumes 50 contiguous features per channel; the vector is ordered as time, then spectrum, then wavelet, then pairwise connectivity. The accuracy drop from 0.98 to 0.82 was measured by zeroing a large slice of that vector at inference time, without retraining.
3. **Task 9 PSNR uses the noisy waveform as the reference.** The reported 10.62 dB measures how far the denoised signal moved from the noisy input. Fidelity to the clean recording was not computed. The classifier trained on denoised features does have a real validation report: accuracy 0.87, balanced accuracy 0.886, ROC-AUC 0.973.
4. **Task 10 did not match the EEG distribution.** Discriminator loss sits near 0 after epoch 10. Real standard deviation is about 0.02; synthetic standard deviation is about 0.92. The downstream validation score (accuracy 0.98, balanced accuracy 0.970) was computed after each feature matrix was z-scored on its own, so it is not evidence that the synthetic recordings look like EEG.
5. **Shared evaluation issues.** `StandardScaler` is fit separately on train, validation, and test. The checkpoint line `state_dict().copy()` keeps references to the live weights; Task 10 shows the logged “best” epoch (balanced accuracy 0.989) was not the epoch that was evaluated (0.970, the last epoch). Sampling rate is 256 Hz in the feature code and 500 Hz in the Task 5 spectrogram.

Task 4’s statistics and plots match the brief. The class commentary there is based on one file per class, including a near-flat channel and one very large spike, so those channel stories are file-level observations.

The same review lists the code defects themselves, with the file and the line that is wrong, in [review/post-hackathon-audit.md](review/post-hackathon-audit.md) under “Bugs in the code”. The ones that change a result are:

| Where | Bug |
|-------|-----|
| `Task 6/building-the-baseline-model.ipynb` | `fft(data, axis=1)` transforms across the 19 channels. Time is the last axis. Both features are then averaged to one number per recording. |
| `Task 7`, `Task 9`, `Task 10`, `Utility Scripts/advanced-feature-extractor.ipynb` | `StandardScaler` is fit inside feature extraction, so each split is scaled with its own mean and variance. |
| `Task 7`, `Task 9`, `Task 10` training loops | `best_model = model.state_dict().copy()` stores the live parameter tensors. Later epochs overwrite the saved “best” weights. |
| `Task 7` and the feature-extractor notebook | Hjorth complexity omits division by mobility. The feature named coherence is `np.correlate(...).max()`, not magnitude-squared coherence. A flat channel divides by zero in mobility and in `corrcoef`. |
| `Task 5/extracting-frequency-domain-features.ipynb` | FFT axis is cycles/sample labeled as Hz. Spectrogram `fs=500` disagrees with `fs=256` elsewhere. Detail-level titles do not match `wavedec` order. `np.log(Sxx)` has no floor. |
| `Task 8/interpretability-of-the-best-model.ipynb` | Channel scores are a reshape of the 512-d gated vector. Masking zeros `50` feature indexes per “channel”, which is not how the 950-vector is laid out. `evaluate_model` is called and is not defined in that notebook. |
| `Task 9/denoising.ipynb` | PSNR is called as `calculate_psnr_eeg(noisy, denoised)`. The peak is `max(abs(noisy))`. |
| `Task 10/generative-modeling-techniques-for-synthetic-eeg.ipynb` | Class labels for the downstream classifier are cast to `float32`. `generate_synthetic_samples` reads `generator.num_classes`, which the generator never sets. The frequency-band modules are unused. |

## What I would do with the data

These are the experiments I would run if a lab has, or can share, an EEG corpus. They do not require this specific hackathon set.

- Fit the scaler on training features only, and choose checkpoints with a training criterion or a nested split so the reported validation number is untouched.
- Replace the Task 6 baseline with per-channel band power and per-channel zero-crossing rate, with balanced class weights, so the deep model is compared to a real baseline.
- Score denoising as PSNR and spectral error against the clean signal, then retrain.
- Rank channels by occlusion or permutation of the 19 raw channels, retrain without the top channels, and compare that drop to a random-channel control.
- Judge a generator by power-spectrum distance and channel-correlation error before quoting a downstream accuracy.
- Report confidence intervals, especially for class 3, and prefer a smaller model if the linear feature model is already close.

## How to cite the numbers

Quote a metric only if it appears in the matching notebook output, and pair the Task 7 numbers with the audit above. I would describe the project as a hackathon system plus a post-hoc review, not as a validated clinical model.
