# Post-hackathon audit

Review of the executed notebooks in this repository. No recording files were available, so no cell was re-run and no metric was replaced. “Reported” means a value printed in a notebook output.

## Task 4 — time-domain statistics

The loader prints shape `(5608, 19, 500)`. For the first file of each class the notebook draws 19 single-channel plots and one overlay, and computes mean, zero-crossing rate, peak-to-peak range, energy (`sum of squares`), RMS, and population variance. Those formulas match the usual definitions. Zero-crossing rate is the number of sign changes divided by 500.

The class write-up generalizes from that one file. In the complex-partial example, channel 11 has energy 24.6 and channel 16 is essentially flat. Those belong in a per-file note. A class statement needs the distribution over the folder.

## Task 5 — spectra and wavelets

`pywt.wavedec` is called with `db4` and `level=4` on one recording per class, and a spectrogram is drawn for all 19 channels. Three presentation errors:

- `np.fft.fftfreq(500)` is in cycles per sample (about −0.5 to 0.5). The axis label says Hz, and the negative half is included.
- The spectrogram uses `fs = 500`. Feature extraction later uses 256 Hz. One of those rates is wrong for this dataset; both cannot be right.
- `wavedec` returns `[cA4, cD4, cD3, cD2, cD1]`. The plot titled “Detail Level 1” is `coeffs[1]`, which is `cD4`, the coarsest detail.

Similarity of a coefficient block to the original signal was judged visually. Coefficient length is much shorter than 500 samples, so the traces are not on the same grid. A quantitative check reconstructs each level to the original length and reports correlation.

## Task 6 — baseline

```python
fourier_transform = np.abs(fft(data, axis=1))  # data is (N, 19, 500)
fourier_features = fourier_transform.mean(axis=(1, 2))
zero_crossing_rates = compute_zero_crossing_rate(data).mean(axis=1)
```

Axis 1 is the channel axis. The spectrum of the time series is along the last axis. Both features are then reduced to one scalar per recording. The SVM is `SVC(kernel="linear", probability=True)` with no class weight.

Reported validation result: accuracy 0.67, ROC-AUC 0.770, balanced accuracy 0.490. Class 2 precision and recall are 0.00 (support 137). This is a faithful evaluation of a two-number feature. It is not a spectral baseline.

## Task 7 — main classifier

Feature layout, which later tasks also assume:

| Block | Count | Layout |
|-------|------:|--------|
| Time domain | 209 | 11 stats × 19 channels |
| Welch bands + spectral edge | 114 | 6 × 19, `fs=256` |
| db4 wavelet stats | 285 | 5 coefficient blocks × 3 stats × 19 |
| Pairwise connectivity | 342 | 171 pairs × (correlation, peak of `np.correlate`) |
| Total | 950 | concatenation in that order |

Reported validation result: accuracy 0.98, balanced accuracy 0.9876, ROC-AUC 0.9983, with the per-class table in the root README. Parameter count 1,445,764 matches a manual sum of the PyTorch parameters, including batch-norm scale and bias. Batch-norm running statistics are buffers, so they correctly stay out of `parameters()`.

Issues that qualify the number:

- `combine_features` fits a new `StandardScaler` on whatever array it is given. Train, validation, and the 779 test files are scaled independently. `test_outputs.csv` was produced under test-set scaling.
- Training keeps `best_model = model.state_dict().copy()`. `state_dict()` returns the parameter tensors themselves. A shallow copy still points at weights that later epochs overwrite. Task 10 is the demonstration: the log’s best balanced accuracy is 0.9889 at epoch 27, and the evaluated model matches epoch 50 (0.9696).
- The kept checkpoint is the one with the highest validation balanced accuracy. The published validation score is the quantity that was used for selection.
- Hjorth complexity in the code is `std(second difference) / std(first difference)`. The standard definition also divides by mobility of the signal. The connectivity feature called coherence is `abs(np.correlate(sig_i, sig_j)).max()`, which is an unnormalized lag peak.
- 1.45 million parameters on 5,608 samples is large relative to the “fewer parameters” line in the brief. The feature set, not the depth, is the plausible reason the classes separate.

`test_outputs.csv` has 779 unique `test_*.npy` names and labels in `{0,1,2,3}`. Predicted counts are 410 / 296 / 67 / 6 (52.6% / 38.0% / 8.6% / 0.8%), against a training prior of 49.6% / 39.2% / 9.7% / 1.5%. Class 3 is under-predicted. The file has no probabilities.

## Task 8 — channels

`AttentionBlock` runs on the 512-dimensional hidden vector, after `Linear(950, 512)`. The hook stores the module output, which is the gated hidden vector. `512 % 19 != 0`, so the code pads one zero and splits the vector into 19 groups. Those groups are not channels.

Masking uses `features_per_channel=50`. Fifty contiguous indexes do not correspond to one channel, because the 950-vector is blocked by feature family (209, then 114, then 285, then 342). The union of the four “top 3” index sets zeros 450 features at validation time. The model is not retrained. Reported accuracy falls from 0.98 to 0.82, balanced accuracy from 0.9876 to 0.8266, ROC-AUC from 0.9983 to 0.9813.

Recall before and after that mask:

| Class | Before | After |
|-------|-------:|------:|
| Normal | 0.99 | 0.67 |
| Complex partial | 0.97 | 0.98 |
| Electrographic | 0.99 | 0.94 |
| Video-detected | 1.00 | 0.71 |

The Task 8 README describes complex-partial and electrographic performance as the sharper drops, and video-detected as the smaller one. The table above has the opposite pattern for recall. Names `C1`–`C19` in that README are indexes into this slicing, not 10–20 electrode labels. The dataset interface in these notebooks does not include a montage.

## Task 9 — denoising

`MultiChannelEEGDenoiser` uses `sym4`, five levels, soft thresholding, and a MAD noise estimate on each detail band. Approximation coefficients are kept. The run completed for `5608 × 19` channels.

PSNR is `calculate_psnr_eeg(X_train_noise, denoised_data)`. The reference is the noisy input. An unchanged signal would score as infinite. Reported mean is 10.62 dB (per-channel means about 10.4–11.2 dB, min 7.36, max 25.90). That number is a change score. Clean-versus-denoised PSNR was not computed.

The classifier trained on features of the denoised set, and evaluated on the existing clean validation feature file, reports accuracy 0.87, balanced accuracy 0.8857, ROC-AUC 0.973. The best logged balanced accuracy is epoch 50, which is the last epoch, so the checkpoint-copy bug does not change which weights were kept. Validation loss rose from 0.36 at epoch 9 to 0.65 at epoch 50.

## Task 10 — generator

The trained generator is a conditional stack: label embedding, linear projection, two stride-2 transposed convolutions, tanh, output `(batch, 19, 500)`. `EEGFrequencyEncoder` and `PhysiologicalConstraints` are defined and never called.

Training log: discriminator loss is about 0 from epoch 10 through epoch 100, generator loss stays between about 6 and 15. Printed amplitude comparison:

| Class | Wasserstein | Real std | Synthetic std | Synthetic mean |
|-------|------------:|---------:|--------------:|---------------:|
| Normal | 0.914 | 0.017 | 0.917 | 0.209 |
| Complex partial | 0.949 | 0.032 | 0.961 | 0.150 |
| Electrographic | 0.881 | 0.021 | 0.905 | −0.137 |
| Video-detected | 0.893 | 0.026 | 0.924 | 0.101 |

Real means are about 0. The classifier trained on synthetic features still reports validation accuracy 0.98 and balanced accuracy 0.9696. Both feature matrices were standardized on their own split, which removes the scale gap before learning. The evaluated balanced accuracy matches epoch 50, not the logged peak of 0.9889 at epoch 27. The notebook does not contain a waveform or spectrogram comparison of synthetic and real signals.

## Bugs in the code

These are defects in the source, separate from the interpretation in the task READMEs. The notebooks were not re-run. A bug is listed only when the code does something other than what the surrounding comment or the problem statement describes.

### Baseline features — `Task 6/building-the-baseline-model.ipynb`

`compute_fourier_transform` calls `fft(data, axis=1)` on arrays of shape `(samples, 19, 500)`. Axis 1 is the channel axis, so the transform is a 19-point FFT across electrodes. The time axis is axis 2. The next cell then does `.mean(axis=(1, 2))`, which collapses that array to one float per recording.

`compute_zero_crossing_rate` takes `mean` over axis 1 of a `(samples, 19, 499)` sign-change array, leaving a value at every time index, and the caller averages those as well. The SVM is fit on two columns.

`train_test_split` is imported and never used. The split itself comes from the organizer folders, which is what the brief required.

### Shared feature code

The same functions are in `Task 7/building-the-best-model.ipynb` and `Utility Scripts/advanced-feature-extractor.ipynb`. Task 9 and Task 10 import the utility notebook.

- **Scaler fit on the batch it is transforming.** `combine_features` builds a new `StandardScaler` and calls `fit_transform` on that call’s array. Train, validation, and test each get their own mean and variance. There is no saved scaler.
- **Hjorth complexity is incomplete.** The code sets `complexity = std(second difference) / std(first difference)`. Standard Hjorth complexity is that quantity divided by mobility, where `mobility = std(first difference) / std(signal)`.
- **Division by zero on a flat channel.** Mobility divides by `std(channel_signal)`. Complexity divides by `std(first difference)`. `np.corrcoef` on a constant channel returns NaN. Task 4’s complex-partial example has a channel whose energy prints as 0, so this path is reachable. A finite training loss only shows that it did not destroy the whole run.
- **Spectral edge.** `total_power / total_power[-1]` divides by zero when the Welch spectrum is all zeros, and `np.where(total_power >= 0.95)[0][0]` indexes an empty array when the cumulative sum never reaches 0.95.
- **“Coherence” is an unnormalized lag peak.** `np.abs(np.correlate(sig_i, sig_j)).max()` uses the default full correlation. It is not magnitude-squared coherence, and its scale follows amplitude.
- **Sampling rate is a literal `256`.** Nothing in the loader reads a rate from the file. Task 5’s spectrogram hardcodes `fs = 500` instead.

### Spectra — `Task 5/extracting-frequency-domain-features.ipynb`

- `np.fft.fftfreq(n)` is plotted with the label “Frequency (Hz)”. Without multiplying by the sampling rate the unit is cycles per sample, and the plot includes the negative frequencies.
- Wavelet panel titles use `Detail Level {j}` for `coeffs[j]`. For `level=4`, `coeffs[1]` is `cD4`, not detail level 1. `wavedec` order is `[cA4, cD4, cD3, cD2, cD1]`.
- The per-class figure plots only `coeffs[0]` (the approximation) in the wavelet column. Full approximation and detail traces are drawn for `X[0]` only, which is a single recording of one class.
- `np.log(Sxx)` is passed straight to `pcolormesh`. A zero bin becomes `-inf`.

### Checkpointing — Task 7, Task 9, and Task 10

All three training loops do `best_model = model.state_dict().copy()` and later `model.load_state_dict(best_model)`. `state_dict()` returns the parameter tensors in the module. `dict.copy()` is shallow, so the stored tensors are the same objects the optimizer updates. The restored “best” model is the last epoch unless those tensors were cloned.

Task 10 is the executed evidence. The log saves a new best at epoch 27 with balanced accuracy 0.9889. The evaluation cell prints 0.9696, which is the epoch-50 line in the same log.

### Explainability — `Task 8/interpretability-of-the-best-model.ipynb`

- The hook is registered on `AttentionBlock` and reads `output`. `forward` returns `x * sigmoid_gate`, the gated 512-vector, not the gate.
- `get_attention_per_channel` does `hidden_dim // 19`. For 512 that remainder is 18, so it pads one zero and reshapes to `(batch, 19, 27)`. The first linear layer was `Linear(950, 512)`, which has already mixed every feature. The 19 slices are not channels.
- `mask_all_top_channels(..., features_per_channel=50)` zeros `[50 * k, 50 * k + 50)`. The 950-vector is 209 time features, then 114 spectral, then 285 wavelet, then 342 pairwise features. Index `50 * k` is not channel `k`.
- `find_top_channels_per_class` groups rows by `argmax` of the logits, so the average is over predicted class, not the label.
- The validation cell calls `evaluate_model`, and that function is not defined anywhere in this notebook. The saved output exists, so the name was already in the Kaggle kernel. A fresh run of this file fails at that call.

### Denoising — `Task 9/denoising.ipynb`

- `calculate_psnr_eeg` is called with `(X_train_noise, denoised_data)`. The error inside is the gap between the denoised signal and the noisy signal. The peak in the formula is `max(abs(original))` on that same noisy channel. `mse == 0` returns `+inf`.
- `compute_threshold(coeffs[i], i)` passes the list index as the level. In `wavedec` order, index 1 is the coarsest detail and the last index is the finest, so the multiplier `(1 + i / self.level)` is largest on the finest band. The argument name `level` does not match the wavelet level number.
- `butter` and `filtfilt` are imported and never called.

### Generator — `Task 10/generative-modeling-techniques-for-synthetic-eeg.ipynb`

- `EEGFrequencyEncoder` and `PhysiologicalConstraints` are never instantiated. `PhysiologicalConstraints.forward` would also call `filtfilt` on a detached NumPy array, which blocks gradients, and its gamma cutoff is 100 Hz.
- `generate_synthetic_samples`, when `class_distribution` is omitted, samples with `generator.num_classes`. `ConditionalGenerator` never assigns `num_classes`. That branch raises `AttributeError`. The executed cell passed a distribution, so this line did not run.
- The downstream training cell builds labels with `torch.tensor(synthetic_labels, dtype=torch.float32)` and then uses `CrossEntropyLoss`, which expects integer class indexes (`torch.long`). The notebook still contains a finished 50-epoch log, so this source and that log do not agree; a clean re-run should cast the labels with `.long()` before the loss.
- Real recordings are given to the discriminator as loaded, with standard deviation about 0.02. The generator ends in `tanh`, so its outputs lie in `[-1, 1]`. There is no amplitude matching between the two.

### What is not a bug

The Task 7 parameter count matches the module. Train and validation folders are loaded separately. `test_outputs.csv` rows line up with `os.listdir` order because the test loader uses `shuffle=False` and filenames were appended in that same order. The denoiser’s future list is consumed in submission order, so channel results are written to the right index.

## Reading order for someone short on time

1. Task 7 validation cell and the feature functions above it.
2. This note, especially “Bugs in the code”, then the Task 6, Task 8, Task 9, and Task 10 sections.
3. `test_outputs.csv` only as the submitted label file, not as a scored test result.
