# Change log

All notable changes to this project will be documented in this file.

## [0.3.0] 2024-04-18
### Changed
- Update dependencies with `torchoutil`, and clean a lot of dead code.
- Convnext model (BL version) is now automatically selected during HDF preparation. ([#5](https://github.com/Labbeti/conette-audio-captioning/issues/5))

### Fixed
- Requirements versions specified during installation. ([#4](https://github.com/Labbeti/conette-audio-captioning/issues/4))
- Preparation now correctly loads `scale_layers` in ConvNext during preparation. ([#5](https://github.com/Labbeti/conette-audio-captioning/issues/5))
- ConvNeXt 75 iterations is now loaded instead of ConvNeXt 70 iterations for CoNeTTE.

## [0.2.2] 2024-01-15
### Added
- Multiple candidates, predictions and probabilities in model outputs.
- `train_and_enable_grad` method in `CoNeTTEModel` class.

### Changed
- Rename `eval_and_detach` to `eval_and_disable_grad` in `CoNeTTEModel` class.

## [0.2.1] 2024-01-12
### Added
- `conette-predict` now support CNext-trans (baseline) model.

## [0.2.0] 2024-01-12
### Added
- CoNeTTE training source code, with entire data processing.
- ConvNeXt-trans baseline training source code, with entire data processing.
- ConvNeXt tag logits to CoNeTTE model outputs during inference.

## [0.1.4] 2023-11-20
### Fixed
- Fix forbid repetition mode argument.

## [0.1.3] 2023-11-20
### Added
- Forbid repetition mode argument to LightningModule and HuggingFace wrapper.

## [0.1.2] 2023-11-17
### Fixed
- Task embeddings inputs `wavcaps_audioset_sl` and `wavcaps_bbc_sound_effects`.

## [0.1.1] 2023-11-09
### Added
- Unittests for hf model.

### Fixed
- Fix sample path for PyPI package.

## [0.1.0] 2023-11-09
### Added
- First version of the conette package to load CoNeTTE model.
