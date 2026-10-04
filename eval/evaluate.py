#!/usr/bin/env python3
"""Score the model on a test manifest and report CER and WER.

Manifest is JSON Lines with "audio_path" and "text" per line. An optional "src"
field splits the report into subsets.

    python eval/evaluate.py --manifest eval/fleurs_ne_test.jsonl
    python eval/evaluate.py --manifest test.jsonl --device cpu --out results.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

PUNCT = re.compile(r"[।,.?!;:\"'’‘“”\-–—()\[\]{}/\\|]")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", PUNCT.sub(" ", text)).strip()


def edit_distance(ref: list, hyp: list) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def score(refs: list[str], hyps: list[str]) -> dict:
    word_err = word_tot = char_err = char_tot = 0
    for r, h in zip(refs, hyps):
        word_err += edit_distance(r.split(), h.split())
        word_tot += len(r.split())
        char_err += edit_distance(list(r.replace(" ", "")), list(h.replace(" ", "")))
        char_tot += len(r.replace(" ", ""))
    return {"n": len(refs),
            "cer": round(char_err / max(char_tot, 1), 4),
            "wer": round(word_err / max(word_tot, 1), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="BlackfireAI/nepali-asr-xlsr-300m")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", help="write full results and hypotheses here")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(a.threads)
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForCTC.from_pretrained(a.model).to(device).eval()

    manifest = Path(a.manifest)
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    # Relative audio paths are resolved against the manifest, not the shell's working
    # directory, so the benchmark runs the same from anywhere in the repo.
    for row in rows:
        if not Path(row["audio_path"]).is_absolute():
            row["audio_path"] = str((manifest.parent / row["audio_path"]).resolve())
    print(f"{len(rows)} utterances on {device}", flush=True)

    refs, hyps, srcs = [], [], []
    audio_seconds = 0.0
    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import torchaudio.functional as AF
                audio = AF.resample(torch.from_numpy(audio), sr, 16000).numpy()
            audio_seconds += len(audio) / 16000
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            logits = model(inputs.input_values.to(device)).logits
            hyp = processor.batch_decode(logits.argmax(-1).cpu().numpy())[0]
            refs.append(normalise(row["text"]))
            hyps.append(normalise(hyp))
            srcs.append(row.get("src", "all"))
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}  {time.time() - start:.0f}s", flush=True)

    elapsed = time.time() - start
    results = {"model": a.model, "manifest": a.manifest, "device": device,
               "decode": "greedy_ctc", "overall": score(refs, hyps),
               "rtf": round(elapsed / audio_seconds, 4),
               "realtime_factor": round(audio_seconds / elapsed, 1)}
    for s in sorted(set(srcs)):
        if s != "all":
            keep = [i for i, v in enumerate(srcs) if v == s]
            results[s] = score([refs[i] for i in keep], [hyps[i] for i in keep])
    print(json.dumps(results, indent=1))

    if a.out:
        payload = {"results": results,
                   "hypotheses": [{"src": s, "ref": r, "hyp": h}
                                  for s, r, h in zip(srcs, refs, hyps)]}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
