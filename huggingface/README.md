---
language:
- ne
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- nepali
- devanagari
- wav2vec2
- xls-r
- ctc
base_model: facebook/wav2vec2-xls-r-300m
datasets:
- google/fleurs
metrics:
- cer
- wer
model-index:
- name: nepali-asr-xlsr-300m
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: FLEURS Nepali
      type: google/fleurs
      config: ne_np
      split: test
    metrics:
    - type: cer
      value: 0.1198
      name: Character Error Rate
    - type: wer
      value: 0.3744
      name: Word Error Rate
---

<img src="assets/blackfire.svg" alt="Blackfire A.I." width="260">

# Nepali ASR (XLS-R 300M)

Speech recognition for Nepali. A fine-tune of
[facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
on 611 hours of Nepali speech, producing Devanagari text.

316M parameters, 16 kHz mono input, CTC decoding with no language model.
Runs at 14x realtime on CPU.

## Install

```bash
pip install torch==2.13.0 torchaudio==2.11.0 transformers==5.17.0 soundfile==0.14.0
```

Those are the versions this model was tested with. Newer ones will most likely work.
No GPU needed, it is fast on CPU.

## Usage

```python
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

name = "BlackfireAI/nepali-asr-xlsr-300m"
processor = AutoProcessor.from_pretrained(name)
model = AutoModelForCTC.from_pretrained(name).eval()

audio, sr = sf.read("audio.wav", dtype="float32")   # must be 16 kHz mono
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values).logits
print(processor.batch_decode(logits.argmax(-1).numpy())[0])
```

## Results

FLEURS Nepali test split, 400 utterances, held out and never trained on. Greedy CTC,
punctuation stripped, NFC normalised.

| metric | value |
|---|---:|
| CER | 0.1198 |
| WER | 0.3744 |
| realtime factor, CPU 8 threads | 14.4x |

Scored in the same run against public Nepali ASR models on the same 400 utterances:

| model | CER | WER |
|---|---:|---:|
| this model | 0.120 | 0.374 |
| anish-shilpakar/wav2vec2-nepali | 0.160 | 0.483 |
| gagan3012/wav2vec2-xlsr-nepali | 0.225 | 0.672 |
| spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr | 0.252 | 0.837 |

Three further models returned CER at or above 1.0 on this set and are excluded as
non-functional rather than weak.

## Limitations

- No punctuation or capitalisation. The vocabulary is 68 Devanagari characters plus
  separator, pad and unknown.
- Devanagari only. It cannot transcribe English, and fails on code-switched Nepali and
  English speech (CER 0.626 on a 1,737 utterance code-switched set).
- Numbers are output as words, not numerals.
- No built in segmentation. Chunk long audio with a VAD first.
- Trained mostly on read, broadcast and narrated speech. Spontaneous conversation is
  harder.

## Training data

611 hours of Nepali speech. Principal sources: IndicVoices (378.4 h), OpenSLR 54 and 43
(81.5 h), IndicVoices-R (49.9 h), internal recordings (42.2 h), Vaani (27.2 h),
WorldSpeech (12.9 h) and the FLEURS train split (1.2 h).

The FLEURS test split was held out.

## Licence

CC BY-NC-SA 4.0. Non-commercial use only.

Base model facebook/wav2vec2-xls-r-300m is Apache 2.0.

## Citation

```bibtex
@misc{blackfire2026nepaliasr,
  title  = {Nepali ASR: an XLS-R 300M fine-tune for Devanagari Nepali},
  author = {Blackfire A.I. Pvt. Ltd.},
  year   = {2026},
  url    = {https://github.com/BlackfireAI/nepali-asr-xlsr}
}
```

Code, evaluation scripts and reproducible benchmarks:
[github.com/BlackfireAI/nepali-asr-xlsr](https://github.com/BlackfireAI/nepali-asr-xlsr)
