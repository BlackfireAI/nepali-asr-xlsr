<img src="assets/blackfire.svg" alt="Blackfire A.I." width="260">

# Nepali ASR (XLS-R 300M)

Speech recognition for Nepali. A fine-tune of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
on 611 hours of Nepali speech, producing Devanagari text.

316M parameters, 1.2 GB, 16 kHz mono input, CTC decoding with no language model.
Runs at 14x realtime on CPU.

## Install

Python 3.9 or newer. A GPU is optional, this model is fast on CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That installs the exact versions this model was tested with:

```
torch 2.13.0 · torchaudio 2.11.0 · transformers 5.17.0 · soundfile 0.14.0
```

Newer versions will most likely work. If something behaves oddly, pin to these first.

Check it works:

```bash
python transcribe.py your_audio.wav
```

The model downloads from Hugging Face on first use, about 1.2 GB, and is cached
afterwards.

## Use

```bash
python transcribe.py audio.wav
python transcribe.py --device cpu one.wav two.wav
python transcribe.py --json out.json recordings/*.wav
```

From Python:

```python
from transcribe import NepaliASR

asr = NepaliASR()
print(asr("audio.wav"))
```

Or with transformers directly:

```python
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

name = "BlackfireAI/nepali-asr-xlsr-300m"
processor = AutoProcessor.from_pretrained(name)
model = AutoModelForCTC.from_pretrained(name).eval()

audio, sr = sf.read("audio.wav", dtype="float32")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values).logits
print(processor.batch_decode(logits.argmax(-1).numpy())[0])
```

Audio is resampled to 16 kHz and downmixed to mono by `transcribe.py`. If you call the
model directly, resample first.

## Demo

Real output from this model on three FLEURS test clips, unedited. Reproduce with
`eval/fetch_fleurs.py` then `python transcribe.py eval/fleurs_ne_audio/fleurs_00002.wav`.

**fleurs_00002, 13.0 s. Correct.**

```
reference  एक क्षेत्रको यात्रा भर्चुअल रूपमा साझा गर्नु पनि यात्रा प्रतिबिम्बित गर्ने र
           भविष्यका कक्षाहरूसँग अनुभव साझा गर्नेका लागि एक राम्रो तरिका हो
output     एक क्षेत्रको यात्रा भर्चुवल रूपमा साझा गर्नु पनि यात्रा प्रतिबिम्बित गर्ने र
           भविष्यका कक्षहरूसँग अनुभव साझा गर्नेका लागि एक राम्रो तरिका हो
```

Two character level differences, `भर्चुअल` against `भर्चुवल` and `कक्षाहरू` against
`कक्षहरू`. Both are orthographic variants of the same words.

**fleurs_00000, 9.6 s. Shows the digit limitation.**

```
reference  सेनाको आगमन हुनुअघि हाइटीले सन् 1800 को दशकपछि यो रोगसँग सम्बन्धित
           समस्याहरूको सामना गर्नुपरेको थिएन
output     सेनाको आगमन हुनु अघि हाइटिलेसन अठाएरचयको दशक पछि यो रोगसँग सम्बन्धित
           समस्याहरूको सामना गर्नुपरेको थिएन
```

The second half is correct. The model renders `1800` phonetically rather than as digits,
and runs `हाइटीले सन्` together. Numerals are the weakest part of the output.

**No punctuation in any of the above.** That is expected. See Limitations.

## Benchmarks

### FLEURS Nepali, 400 utterances

FLEURS is an independent read-speech corpus built by a separate project. The test split
was held out and never trained on, so this is an out of domain measurement.

Measured with the script in this repo, greedy CTC, punctuation stripped, NFC normalised:

```bash
pip install datasets
python eval/fetch_fleurs.py
python eval/evaluate.py --manifest eval/fleurs_ne_test.jsonl --device cpu
```

| metric | value |
|---|---:|
| CER | **0.1198** |
| WER | **0.3744** |
| realtime factor, CPU 8 threads | **14.4x** |

Full output including per utterance hypotheses is in `eval/results_fleurs.json`.

### Against public Nepali models

Every model below was scored by us in the same run, on the same 400 FLEURS utterances,
with the same normalisation and greedy decoding. Reproduce with `eval/evaluate.py` and
the model id in the first column.

| model | CER | WER | realtime factor |
|---|---:|---:|---:|
| **this model** | **0.120** | **0.374** | 13.1x |
| anish-shilpakar/wav2vec2-nepali | 0.160 | 0.483 | 8.6x |
| gagan3012/wav2vec2-xlsr-nepali | 0.225 | 0.672 | 8.6x |
| spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr | 0.252 | 0.837 | 8.2x |

`gagan3012/wav2vec2-xlsr-nepali` is the most downloaded Nepali ASR model on the Hub, at
over 400,000 downloads.

Two models were tested and excluded as non-functional on this set rather than merely weak:
`Harveenchadha/vakyansh-wav2vec2-nepali-nem-130` returned CER above 1.0, and two Whisper
Nepali fine-tunes returned CER 1.0. One further model,
`shniranjan/wav2vec2-large-xlsr-300m-nepali`, could not be loaded without the optional
`pyctcdecode` dependency and was not scored.

Raw per-model output is in `eval/results_public_fleurs.json`.

### Generalisation

| test set | CER | WER |
|---|---:|---:|
| internal held-out, 400 | 0.117 | 0.395 |
| FLEURS, 400, independent | 0.120 | 0.374 |

Word error is slightly lower on the independent set than on the in domain one, so the
model is not fitted to its own test distribution.

### Speed on CPU

No GPU required. A single CTC forward pass, no autoregressive decoding and no language
model, so throughput is predictable and scales with threads.

Measured on 20 FLEURS clips, 234 seconds of audio. Realtime factor is seconds of audio
transcribed per second of wall clock, so 13.1x means a one hour recording takes about
four and a half minutes.

| CPU threads | wall clock | realtime factor | RTF |
|---:|---:|---:|---:|
| 1 | 76.3 s | 3.1x | 0.326 |
| 2 | 42.5 s | 5.5x | 0.182 |
| 4 | 28.2 s | 8.3x | 0.120 |
| 8 | 26.1 s | 9.0x | 0.111 |
| 16 | 17.9 s | **13.1x** | 0.076 |

Even on a single thread it runs at 3x realtime, so batch transcription is practical on a
laptop. Gains flatten between 4 and 8 threads. If you are sizing a deployment, 4 threads
per worker is the efficient point.

Set the thread count before loading the model:

```python
import torch
torch.set_num_threads(4)
```

## Limitations

**No punctuation or capitalisation.** The vocabulary is 68 tokens: Devanagari characters,
word separator, pad and unknown. Output is unpunctuated running text.

**Devanagari only. It cannot transcribe English.** On code-switched Nepali and English
speech it fails, because Latin characters are not in the vocabulary at all. Measured on a
1,737 utterance code-switched set: CER 0.626. If you need code-switching, this is the wrong
model.

**No digits.** Numbers are written as words, not numerals. `२५००` in a reference will be
transcribed as `दुई हजार पाँच सय`. Account for this when scoring.

**No built in segmentation.** Long audio should be chunked with a VAD before transcription.
There is no hard length cap, but memory grows with input length.

**Read and narrated speech.** The training corpus is predominantly read, broadcast and
narrated speech. Spontaneous conversation is harder and error rates will be higher.

**Orthographic variants.** Nepali spelling variation (ी against ि, श against स, ब against व)
accounts for roughly 4 percent of word error. Some of this is unwinnable because the
reference transcripts themselves are inconsistent.

## Training data

611 hours of Nepali speech. Principal sources:

| hours | source |
|---:|---|
| 378.4 | IndicVoices Nepali |
| 81.5 | OpenSLR 54, OpenSLR 43 |
| 49.9 | IndicVoices-R Nepali |
| 42.2 | Internal recordings |
| 27.2 | Vaani Nepali |
| 12.9 | WorldSpeech Nepali |
| 1.2 | FLEURS Nepali (train split only) |

The FLEURS test split was held out and never trained on, so the FLEURS numbers above are
a genuine out of domain measurement.

## Licence

Weights and code are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
**Non-commercial use only.**

The base model, facebook/wav2vec2-xls-r-300m, is Apache 2.0.

## Credits

Built on [XLS-R](https://arxiv.org/abs/2111.09296) (Babu et al., 2021) by Meta AI.

Training data from [IndicVoices](https://ai4bharat.iitm.ac.in/indicvoices),
[OpenSLR 54](https://openslr.org/54/), [OpenSLR 43](https://openslr.org/43/),
[Vaani](https://vaani.iisc.ac.in/) and [FLEURS](https://huggingface.co/datasets/google/fleurs).

## Citation

```bibtex
@misc{blackfire2026nepaliasr,
  title  = {Nepali ASR: an XLS-R 300M fine-tune for Devanagari Nepali},
  author = {Blackfire A.I. Pvt. Ltd.},
  year   = {2026},
  url    = {https://github.com/BlackfireAI/nepali-asr-xlsr}
}
```

Maintained by [Blackfire A.I. Pvt. Ltd.](https://blackfire.com.np)
