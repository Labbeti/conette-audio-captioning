<div align="center">

# CoNeTTE model for Audio Captioning

[![](<https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white>)](https://www.python.org/)
[![](<https://img.shields.io/badge/-PyTorch 1.10.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white>)](https://pytorch.org/get-started/locally/)
[![](https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![](https://img.shields.io/github/actions/workflow/status/Labbeti/conette-audio-captioning/inference.yaml?branch=main&style=for-the-badge&logo=github)](https://github.com/Labbeti/conette-audio-captioning/actions)

</div>

CoNeTTE is an audio captioning system, which generate a short textual description of the sound events in any audio file. The architecture and training are explained in the corresponding [paper](https://arxiv.org/pdf/2309.00454.pdf). The model has been developped by me ([Étienne Labbé](https://labbeti.github.io/)) during my PhD. A simple interface to test CoNeTTE is available on [HuggingFace website](https://huggingface.co/spaces/Labbeti/conette).

## Inference

### Installation
```bash
python -m pip install conette
```

### Usage with python
```py
from conette import CoNeTTEConfig, CoNeTTEModel

config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

path = "/your/path/to/audio.wav"
outputs = model(path)
candidate = outputs["cands"][0]
print(candidate)
```

The model can also accept several audio files at the same time (list[str]), or a list of pre-loaded audio files (list[Tensor]). In this second case you also need to provide the sampling rate of this files:

```py
import torchaudio

path_1 = "/your/path/to/audio_1.wav"
path_2 = "/your/path/to/audio_2.wav"

audio_1, sr_1 = torchaudio.load(path_1)
audio_2, sr_2 = torchaudio.load(path_2)

outputs = model([audio_1, audio_2], sr=[sr_1, sr_2])
candidates = outputs["cands"]
print(candidates)
```

The model can also produces different captions using a Task Embedding input which indicates the dataset caption style. The default task is "clotho".

```py
outputs = model(path, task="clotho")
candidate = outputs["cands"][0]
print(candidate)

outputs = model(path, task="audiocaps")
candidate = outputs["cands"][0]
print(candidate)
```

### Usage with command line
Simply use the command `conette-predict` with `--audio PATH1 PATH2 ...` option. You can also export results to a CSV file using `--csv_export PATH`.

```bash
conette-predict --audio "/your/path/to/audio.wav"
```

### Performance
The model has been trained on AudioCaps (AC), Clotho (CL), MACS (MA) and WavCaps (WC). The performance on the test subsets are :

| Test data | SPIDEr (%) | SPIDEr-FL (%) | FENSE (%) | Vocab | Outputs | Scores |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| AC-test | 44.14 | 43.98 | 60.81 | 309 | [Link](https://github.com/Labbeti/conette-audio-captioning/blob/main/results/conette/outputs_audiocaps_test.csv) | [Link](https://github.com/Labbeti/conette-audio-captioning/blob/main/results/conette/scores_audiocaps_test.yaml) |
| CL-eval | 30.97 | 30.87 | 51.72 | 636 | [Link](https://github.com/Labbeti/conette-audio-captioning/blob/main/results/conette/outputs_clotho_eval.csv) | [Link](https://github.com/Labbeti/conette-audio-captioning/blob/main/results/conette/scores_clotho_eval.yaml) |

This model checkpoint has been trained with focus on the Clotho dataset, but it can also reach a good performance on AudioCaps with the "audiocaps" task.

### Limitations
- The model expected audio sampled at **32 kHz**. The model automatically resample up or down the input audio files. However, it might give worse results, especially when using audio with lower sampling rates.
- The model has been trained on audio lasting from **1 to 30 seconds**. It can handle longer audio files, but it might require more memory and give worse results.

## Train a model
### Requirements
- Intended for Ubuntu 20.04 only. Requires **java** < 1.13, **ffmpeg**, **yt-dlp**, and **zip** commands.
- Recommanded GPU: NVIDIA V100 with 32GB VRAM.
- WavCaps dataset might requires more than 2 TB of disk storage. Other datasets requires less than 50 GB.

### Installation
By default, **only the inference requirements are installed for conette**. To install training requirements you need to use the following command:
```bash
python -m pip install conette[train]
```
If you already installed conette for inference, it is **highly recommanded to create another environment** before installing conette for training.

### Download external models and data
These steps might take a while (few hours to download and prepare everything depending on your CPU, GPU and SSD/HDD).

First, download the ConvNeXt, NLTK and spacy models :
```bash
conette-prepare data=none default=true pack_to_hdf=false csum_in_hdf_name=false pann=false
```

Then download the 4 datasets used to train CoNeTTE :
```bash
cnext_bl_path="$HOME/.cache/torch/hub/checkpoints/convnext_tiny_465mAP_BL_AC.pth"
common_args="data.download=true pack_to_hdf=true audio_t=resample_mean_convnext audio_t.pretrain_path=${cnext_bl_path} post_hdf_name=bl pretag=cnext_bl"

conette-prepare data=audiocaps audio_t.src_sr=32000 ${common_args}
conette-prepare data=clotho audio_t.src_sr=44100 ${common_args}
conette-prepare data=macs audio_t.src_sr=48000 ${common_args}
conette-prepare data=wavcaps audio_t.src_sr=32000 ${common_args} datafilter.min_audio_size=0.1 datafilter.max_audio_size=30.0 datafilter.sr=32000
```

### Train a model
CNext-trans (baseline) on CL only (~3 hours on 1 GPU V100-32G)
```bash
conette-train expt=[clotho_cnext_bl] pl=baseline
```

CoNeTTE on AC+CL+MA+WC, specialized for CL (~4 hours on 1 GPU V100-32G)
```bash
conette-train expt=[camw_cnext_bl_for_c,task_ds_src_camw] pl=conette
```

CoNeTTE on AC+CL+MA+WC, specialized for AC (~3 hours on 1 GPU V100-32G)
```bash
conette-train expt=[camw_cnext_bl_for_a,task_ds_src_camw] pl=conette
```

**About reproducibility** : any training with AC data cannot be reproduced because a part of this data is deleted from the YouTube source, and I cannot share my own audio files.

## Citation
The preprint version of the paper describing CoNeTTE is available on arxiv: https://arxiv.org/pdf/2309.00454.pdf

```bibtex
@misc{labbé2023conette,
	title        = {CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding},
	author       = {Étienne Labbé and Thomas Pellegrini and Julien Pinquier},
	year         = 2023,
	journal      = {arXiv preprint arXiv:2309.00454},
	url          = {https://arxiv.org/pdf/2309.00454.pdf},
	eprint       = {2309.00454},
	archiveprefix = {arXiv},
	primaryclass = {cs.SD}
}
```

## Additional information
- CoNeTTE stands for **Co**nv**Ne**Xt-**T**ransformer with **T**ask **E**mbedding.
- Model weights are available on HuggingFace: https://huggingface.co/Labbeti/conette
- The weights of the encoder part of the architecture is based on a ConvNeXt model for audio classification, available here: https://zenodo.org/record/8020843 under the filename "convnext_tiny_465mAP_BL_AC_70kit.pth".

## Contact
Maintainer:
- Etienne Labbé "Labbeti": labbeti.pub@gmail.com
