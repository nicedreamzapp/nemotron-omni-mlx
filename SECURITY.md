# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report them privately through GitHub: go to the **Security** tab of this repo and click **Report a vulnerability** ([direct link](https://github.com/nicedreamzapp/nemotron-omni-mlx/security/advisories/new)). Only the maintainer sees the report.

Helpful details: what you found, how to reproduce it, which version or commit, your Mac and macOS version, and the input file that triggers it if there is one.

I aim to reply within 3 days and to ship a fix or a clear plan within 30 days, depending on severity. You will be credited in the advisory unless you ask not to be.

## Supported versions

Fixes go into the latest version on `main`. Please update before reporting if you are on an older commit.

## What matters most here

The parts where a mistake could hurt someone:

- model downloads in `scripts/setup_weights.py`, which pull the checkpoint from Hugging Face
- `trust_remote_code=True` in `src/processing.py` and the parity tests, which runs Python code from the `reference/` folder
- the image, video and audio loading in `src/processing.py` and `src/audio.py`, which parse files the user passes in
