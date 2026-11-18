# tests/fixtures - the one reference artefact the checks need

`TINY.BIN` is the released Tiny-2.1 weight file. It is here, and only here,
because the whole verification story rests on it: `tests/ref_tiny.c` is the
original forward pass, `build/gradcheck` measures derivatives against these
weights, and `tools/verify_model.sh` compares answers. Without a fixed
baseline every one of those checks is comparing the code against itself.

381 KB is a test fixture. The rest of `refs/` - 41 MB of old attempts, the
PyTorch trainer, the C# sources - is reference material, and it is **not**
in git. `refs/` is ignored on purpose: it is provenance, read once and
quoted in the documentation, not something a clone has to carry.

## Where the rest comes from

Put the reference tree back at `refs/` if you want to run:

- `tools/ref_gradients.py` (needs `refs/Tiny-2.1-main/training/train2.py`
  and `tiny2.pt`, plus torch) - the float gradient used to confirm the
  reward formulas;
- `tools/build_dataset.py`, which reads
  `refs/Tiny-2.1-main/training/facts/` to build `build/dataset.txt`;
- nothing else. Everything in `src/`, `tests/` and the CI jobs runs from
  `tests/fixtures/TINY.BIN` alone.

The scripts that need it say so and stop; none of them fall back to a
silently different file.

## Licence

Tiny-2.1 is MIT, copyright (c) 2026 WollyMonjac. The full text is in
`TINY.LICENSE.txt`. Three of its training data files are ODbL 1.0 rather
than MIT; `TINY.DATA_LICENSES.md` records which. Neither the data nor the
trainer is redistributed here - only the exported weight file, which the
release ships for exactly this purpose.
