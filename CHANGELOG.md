# Change log

All notable changes to this project will be documented in this file.

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
