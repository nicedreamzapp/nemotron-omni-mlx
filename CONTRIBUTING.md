# Contributing

Thanks for taking a look. This runtime has only been tested on one Mac so far, so a report from
your chip and RAM is useful even if you never send code.

## Reporting a problem

Open an [issue](https://github.com/nicedreamzapp/nemotron-omni-mlx/issues/new/choose). The things
that make it quick to fix:

- your Mac (chip and RAM), macOS version and Python version
- the exact `python -m src.omni ...` command, and whether it was the image, video or audio path
- the failing test name, if a parity test fails
- on a 32 GB Mac, what the `recommendedMaxWorkingSetSize` snippet in the README prints
- the error output, and peak memory if you saw it

## Running it locally

```bash
git clone https://github.com/nicedreamzapp/nemotron-omni-mlx
cd nemotron-omni-mlx
pip install -r requirements.txt
python scripts/setup_weights.py
huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --exclude "*.safetensors" --local-dir reference
python -m src.omni --image photo.jpg --prompt "What's in this picture?"
```

To check the processor with no weights loaded:
`python -m src.omni --prompt "Hi" --dry-run`.

## Tests

The unit tests need no weights. They use tiny random-weight towers and a stub tokenizer, run on
MLX's CPU backend, and are what CI runs on Linux.

```bash
pip install -r requirements-test.txt
pytest
```

The parity tests in `tests/test_*_parity.py` compare against NVIDIA's PyTorch reference. They need
the torch-side deps at the bottom of `requirements.txt`, NVIDIA's code in `reference/` and the bf16
shard 1 in `weights-bf16/`. Without those they skip and say what is missing. If you touch a tower
or the processor, please run them if you can.

## Pull requests

Small and focused is easiest to review. For big changes, open an issue first so we can talk it
through. Say which Mac you tested on, and include the parity numbers if the change touches
`src/vision.py`, `src/audio.py` or `src/processing.py`.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).
