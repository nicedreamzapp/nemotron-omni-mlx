**NVIDIA shipped a 30-billion-parameter model that can see, hear, and talk — and gave the weights away. The catch: the seeing and hearing parts didn't run on a Mac. So I spent an afternoon writing the missing piece.**

Here is the whole thing in thirty-five seconds — the model reading a real cart off my own store, on the laptop, with nothing leaving it.

__VIDEO__

Every couple of weeks I go looking for whatever new open-weight model just dropped, pull it onto my laptop, and see what it can actually do. Most of the time it's a coding model, I run it against the one I already use, and my current favorite wins again. That's a fine result. It's just not much of a story.

This time I found something different.

## What Nemotron Omni actually is

NVIDIA released **Nemotron-3-Nano-Omni-30B-A3B** — a "tri-modal" model, which is a fancy way of saying one brain with eyes and ears attached. You can hand it a picture, a sound file, or a video, and talk to it about what it saw or heard. It's 30 billion parameters total, but only about 3 billion of them fire for any given word, which is why something this capable can run on a laptop at all. The weights are public.

Someone had already done the hard, unglamorous work of shrinking it down to a 4-bit MLX version that fits on Apple Silicon — that's [yayr](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit) over at mlx-community, and this project doesn't exist without that upload. About 19 GB on disk. Ready to go.

Except for one line, buried in the model card:

> The text backbone loads with standard MLX `nemotron_h` tooling. The vision and audio towers require a multimodal runtime that implements the C-RADIO ViT-H and Parakeet Conformer forward passes (e.g. the Evorix on-device engine).

Translated: the brain works on a Mac. **The eyes and ears don't.** The weights for them are right there in the file — all the knowledge, sitting on your disk — but nothing I could find in the open Apple Silicon world knew how to *run* them. NVIDIA's own code for those parts is written for their GPUs. On a Mac it's a locked room with the key visible through the window.

I want to be precise about that parenthetical, because it matters: the card does name an engine, Evorix. I went looking for it — not on Hugging Face, not on GitHub, not anywhere I could find. So as far as I can tell it exists, but not somewhere you or I can go get it. Which leaves the same practical dead end: you download nineteen gigabytes of a model that sees and hears, and on a Mac you can only talk to it.

That's the gap — no *open* runtime for the eyes and ears. Honestly, that's the most interesting kind of thing to find. Not a benchmark to run. A thing that doesn't work yet.

## The part I keep having to relearn

My instinct was that this would take days. Three separate pieces, each one a real port: NVIDIA's vision tower is a ViT with a custom patch generator, their audio side is a Conformer with Transformer-XL relative attention, and then there's the glue that turns a photo into something the language model can read.

It took about half an hour.

Not because I'm fast — because I stopped doing it one piece at a time. I put a separate agent on each tower and let them run at the same time, each one checking its own work against NVIDIA's original code as it went. I keep making the same mistake of estimating this stuff like it's still 2024, and I keep getting corrected by my own laptop.

## Proving it, instead of vibing it

Here's the thing about porting a model: it's very easy to write code that *looks* right, produces numbers, and is quietly wrong. The model doesn't crash. It just gets a little dumber, and you never find out.

So neither tower got to claim victory on vibes. For each one, the test was: run NVIDIA's original PyTorch code and my MLX version **on the same input, with the same weights**, and compare the actual numbers coming out.

| Tower | What was compared | Result |
|---|---|---|
| **Ears** (audio) | Final audio embeddings, 5-second clip | cosine **0.99999** |
| **Eyes** (vision) | Final image embeddings, 448px image | cosine **0.99996** |
| **Eyes**, on CPU math | Same, without GPU shortcuts | **1.00000000** — exact |

That last row is the one I care about. Run it on the CPU, where the math is done precisely, and my port and NVIDIA's are **not close — they're identical.** The tiny gap on the GPU isn't a bug in the port; it's Metal taking shortcuts with float math for speed. Chasing that would be chasing the hardware.

The input processing got the same treatment — every image tile, every audio frame, every token id checked against NVIDIA's reference. 14 tests, token sequences matching exactly, pixels off by less than a millionth.

The brain, meanwhile, needed no porting at all — MLX already understood it. Someone had put `nemotron_h` into mlx-lm before I ever showed up.

## So does it actually work

This is the only part that matters, so here's the first thing I pointed it at once the pieces were connected. Not a test image — a screenshot off my own phone, of my own store, with a cart full of my own products.

I asked: *"What website is this and what is in the cart?"*

> **This is the Divine Tribe website. The cart contains a Gen 2 DC Ceramic Rebuildable Dry Herb Heater, a Replacement Heater Cup, a Wireless Dock Station, 72% Hemp 28% Silk Men's Boxers, and a Quest Lightning Diffuser Kit.**

Every product, correct. When I asked for prices it read all five to the cent — $37.36, $13.12, $80.80, $33.32, $38.67 — and started adding them up. Nine seconds, on a laptop, with nothing leaving it.

Then the ears. I generated a line of speech and handed it the wav:

> **The Divine Tribe vaporizer ships from Humboldt County, California, and this model is running entirely on a MacBook with no Internet connection.**

Word for word, punctuation and all — it even got "Humboldt" right.

The speeds, for anyone keeping score:

| Mode | Speed | Memory |
|---|---|---|
| Text only | **152 tok/s** | 17.9 GB |
| With an image | **67.7 tok/s** | 22.1 GB |
| With audio | **147 tok/s** | 21.0 GB |

The whole thing lives in about 22 GB at its hungriest. That fits on a Mac you can buy today.

And to be clear about what that means: nothing about that cart screenshot or that audio clip left my desk. The model makes no network calls at all — there's no API key, no meter running, no terms of service, and no company on the other end deciding whether I'm allowed to keep doing this tomorrow.

## Two things that fell out along the way

Neither of these was the goal, and both are probably more useful than anything else here.

**NVIDIA's reference code produces NaN on batched audio.** If you feed it two clips of different lengths at once, the shorter one comes back as garbage — not an error, just silent NaN poisoning that spreads through the layers. It's a masking detail: fully-padded rows go to negative infinity, softmax turns that into NaN, and the NaN travels. My port masks differently and stays finite. I want to be careful here — this is one specific path, and I could be wrong about how much it matters in NVIDIA's own pipeline. But it reproduces on their code, not just mine.

**The vision tower doesn't normalize its own input.** There's a normalization layer sitting right there in the checkpoint, and it's dead weight — NVIDIA switches it off and expects whatever calls it to do that job. Feed it raw pixels like a reasonable person would and you get plausible-looking garbage, silently. This one cost real time and it's the trap anyone else attempting this will hit.

**And a third, for the truly nerdy:** several settings in the config file are lies. Not maliciously — they're just vestigial, left over from an earlier design, and the live code ignores them completely. If you build from the config instead of tracing what actually runs, you'll produce something that looks correct and isn't. I only caught it by following the real code path line by line.

## So who does this actually help?

Fair question, and I want to answer it honestly instead of waving at the future.

**The narrow answer: it saves the next person about a week.** Right now, anyone with a Mac who downloads this model reads that line on the model card — *"requires a multimodal runtime that implements the C-RADIO ViT-H and Parakeet Conformer forward passes"* — and that's the end of the road. It's not a warning, it's a wall. It means *go build it yourself*, and most people, reasonably, close the tab. Now they don't have to. Clone it, run it, done. And the three traps I hit are written down, because the normalization one in particular would cost someone a full day of wondering why their model got quietly dumber instead of visibly broken. That's the entire contribution: one person spent the afternoon so nobody else has to spend the week.

I'm not going to pretend that's a huge number of people. Might be a few hundred. Might be twelve. That's fine — twelve people not wasting a week each is still worth an afternoon.

**The wider answer is the one I actually care about.** I do private-AI work for firms that handle other people's confidential material — lawyers, medical practices, accountants. The whole pitch is that the machine doing the reading is the machine on your desk, because a federal court [already ruled](https://nicedreamzwholesale.com/2026/05/22/the-heppner-ruling-warner-v-gilbarco-and-what-confidential-ai-actually-has-to-mean/) that work you hand to a public AI isn't privileged.

Until now, "on your desk" meant text only. If a client sends a photograph of a contract, or a recorded call, or a scan — the private option had nothing to say. You either sent it to somebody's cloud and lost the privilege, or you did it by hand.

That changed today, on my laptop. A model that can *look at* the scanned page and *listen to* the recording, on the machine sitting in front of them, is not a small difference for those people. It's the difference between a tool they can use and a tool they legally can't.

**And the honest third reason:** the gap between "the weights are public" and "you can actually use this" is where most open AI quietly dies. Everyone celebrates the release. Far fewer people do the boring work of making the thing run somewhere real. That gap is usually not a research problem — it's a few hundred lines nobody got around to writing. This one was maybe 90 KB of Python and an afternoon.

That's the part I'd like more people to see. Not that I did something clever — I didn't, I transcribed NVIDIA's own math into a different framework and checked my work. But the wall between an open model and a working model is *thinner than it looks*, and it stays up mostly because everyone assumes someone else will knock it down.

The code is [on GitHub](https://github.com/nicedreamzapp/nemotron-omni-mlx), MIT, with every parity test in it. Don't take my word for any number above — clone it and run the tests on your own Mac.

## Credit where it's due

- **[NVIDIA](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)** built the model and released the weights and reference code openly. None of this happens otherwise.
- **[yayr](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit)** at mlx-community did the 4-bit MLX quantization I built on top of. Go give that upload a like — it deserves more than the zero it has.
- **[Apple's MLX team](https://github.com/ml-explore/mlx)** — and whoever added `nemotron_h` to mlx-lm, which is why the brain needed nothing from me at all.
