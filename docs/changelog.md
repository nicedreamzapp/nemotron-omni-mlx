---
title: Changelog - nemotron-omni-mlx release history
description: History of nemotron-omni-mlx from the git log, from the first pure MLX runtime for Nemotron Omni in July 2026 through the 32 GB memory notes, abliterated builds and README updates.
---

# Changelog

There are no tagged releases yet. This page is built from the commit history on `main`, newest
first. Dates are commit dates.

## 2026-10-09

- README links to [claude-code-local](https://github.com/nicedreamzapp/claude-code-local).
  (`db50311`)

## 2026-10-08

- README footer wording change. (`81319a5`)

## 2026-09-27

- README rewrite: plain summary, demo at the top, a "What I built" section listing each file, and
  corrected setup steps, including fetching `reference/` and the optional extras for video,
  resampling and the parity tests. (`bc0e627`, merged in `9702e3d`)

## 2026-09-19

- README: other MLX routes (mlx-vlm, mlx-audio, NVIDIA's own release) and how to report a parity
  problem. (`fedd022`)

## 2026-09-07

- Added the write-up draft in `article/` (Markdown, HTML, build script, poster and demo video).
  (`0c6755f`)

## 2026-08-25

- README: running the abliterated builds on llama.cpp via GGUF, with the conversion patch, credited
  to [@wacomctl672](https://huggingface.co/wacomctl672). (`80902bf`)

## 2026-08-14

- README: links to abliterated (refusal-removed) MLX builds in 4-bit, 8-bit and bf16. Vision and
  audio towers are left intact. Refusal was removed from blocks 16 and 31 of the language stack.
  (`a16edf0`)

## 2026-08-09

- README: documented the Metal wired-memory ceiling on 32 GB Macs and the `iogpu.wired_limit_mb`
  fix, and moved the warning next to the peak memory numbers. (`859d727`, `2b45048`)

## 2026-07-16

- First release of the runtime: `src/vision.py`, `src/audio.py`, `src/processing.py`,
  `src/omni.py`, `scripts/setup_weights.py` and the parity suite in `tests/`. The commit message
  reports 23/23 parity tests, vision graph-exact on MLX's CPU stream (cosine 1.00000000), audio
  cosine 0.99999130, and exact processor token ids. (`c4188bd`)
- Added the demo video, poster and preview GIF in `docs/`. (`8250dd4`)
- README demo link points to the write-up page, since GitHub shows no player for an mp4 in a repo.
  (`c5df228`)
