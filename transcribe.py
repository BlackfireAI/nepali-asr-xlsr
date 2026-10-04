#!/usr/bin/env python3
"""Transcribe Nepali audio.

    python transcribe.py audio.wav
    python transcribe.py --device cpu one.wav two.wav
    python transcribe.py --model ./model --json out.json *.wav

Any format and sample rate soundfile can read; audio is resampled to 16 kHz and
downmixed to mono automatically.
"""
from __future__ import annotations

import argparse
import json
import sys

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

DEFAULT_MODEL = "BlackfireAI/nepali-asr-xlsr-300m"


class NepaliASR:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForCTC.from_pretrained(model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, audio, sample_rate: int | None = None) -> str:
        if isinstance(audio, str):
            audio, sample_rate = sf.read(audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            import torchaudio.functional as AF
            audio = AF.resample(torch.from_numpy(audio), sample_rate, 16000).numpy()
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.device)).logits
        return self.processor.batch_decode(logits.argmax(-1).cpu().numpy())[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe Nepali speech.")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--json", help="write results to this file instead of stdout")
    a = ap.parse_args()

    asr = NepaliASR(a.model, a.device)
    results = []
    for path in a.audio:
        try:
            text = asr(path)
        except Exception as e:                       # one bad file must not lose the batch
            print(f"{path}: ERROR {e}", file=sys.stderr)
            continue
        results.append({"audio": path, "text": text})
        if not a.json:
            print(f"{path}\t{text}" if len(a.audio) > 1 else text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"{len(results)} transcripts -> {a.json}")


if __name__ == "__main__":
    main()
