# CoNeTTE (ConvNext-Transformer with Task Embedding) for Automated Audio Captioning

This model generate a short textual description of any audio file.

## Installation
```bash
python -m pip install conette
python -m spacy download en_core_web_sm
```

## Usage
```py
from conette import CoNeTTEConfig, CoNeTTEModel

config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

path = "/my/path/to/audio.wav"
outputs = model(path)
candidate = outputs["cands"][0]
print(candidate)
```

The model can also accept several audio files at the same time (list[str]), or a list of pre-loaded audio files (list[Tensor]). In this second case you also need to provide the sampling rate of this files:

```py
import torchaudio

path_1 = "/my/path/to/audio_1.wav"
path_2 = "/my/path/to/audio_2.wav"

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

## Performance

| Test dataset | SPIDEr (%) | SPIDEr-FL (%) | FENSE (%) |
| ------------- | ------------- | ------------- | ------------- |
| AudioCaps | 44.14 | 43.98 | 60.81 |
| Clotho | 30.97 | 30.87 | 51.72 |

This model checkpoint has been trained for the Clotho dataset, but it can also reach a good performance on AudioCaps with the "audiocaps" task.

## Citation
The preprint version of the paper describing CoNeTTE is available on arxiv: https://arxiv.org/pdf/2309.00454.pdf

```
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

The encoder part of the architecture is based on a ConvNeXt model for audio classification, available here: https://huggingface.co/topel/ConvNeXt-Tiny-AT.
More precisely, the encoder weights used are named "convnext_tiny_465mAP_BL_AC_70kit.pth", available on Zenodo: https://zenodo.org/record/8020843.

It was created by [@{{ author }}](https://hf.co/{{author}}).
