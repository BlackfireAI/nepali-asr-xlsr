#!/usr/bin/env python3
"""Download the FLEURS Nepali audio this benchmark scores against.

    pip install datasets
    python eval/fetch_fleurs.py

Writes eval/fleurs_ne_audio/*.wav matching the ids in eval/fleurs_ne_test.jsonl.
The ids are indices into the FLEURS test split, so only the listed subset is written.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

HERE = Path(__file__).resolve().parent
OUT = HERE / "fleurs_ne_audio"
MANIFEST = HERE / "fleurs_ne_test.jsonl"


def main() -> None:
    wanted = {json.loads(l)["id"] for l in open(MANIFEST, encoding="utf-8")}
    OUT.mkdir(exist_ok=True)

    ds = load_dataset("google/fleurs", "ne_np", split="test")
    # Decoding is left to soundfile. Recent datasets versions route audio decoding through
    # torchcodec, which is an extra dependency and resamples; raw bytes avoid both.
    ds = ds.cast_column("audio", Audio(decode=False))

    written = 0
    for i, row in enumerate(ds):
        name = f"fleurs_{i:05d}"
        if name not in wanted:
            continue
        a = row["audio"]
        if a.get("bytes"):
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
        elif a.get("path") and Path(a["path"]).exists():
            wav, sr = sf.read(a["path"], dtype="float32", always_2d=True)
        else:
            continue
        sf.write(OUT / f"{name}.wav", wav.mean(axis=1), int(sr))
        written += 1

    print(f"{written}/{len(wanted)} files written to {OUT}")
    if written < len(wanted):
        print("Some ids were not found. The FLEURS release may have changed since this "
              "manifest was built.")


if __name__ == "__main__":
    main()
