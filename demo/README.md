# Terminal demo

A ~25-second run showing the full loop: training → injected worker failure →
automatic detection → checkpoint restore → resume → completion, with no human
in the loop.

```bash
bash demo/demo.sh
```

[`demo_transcript.txt`](demo_transcript.txt) is a real captured run, committed
so the output is readable without running anything (and so a reviewer can see
it in a diff).

## Why it's the same every time

The failure schedule is a pure function of `(--failure-seed, step)` — see
[Failure injection](../README.md#failure-injection) — so this script injects
exactly **one failure, at step 7, on worker 1**, on every take. That matters
for a demo of a *reproducibility* feature: a recording that showed a different
failure each run would undercut the thing it exists to illustrate.

Worker 1 is the victim rather than worker 0 on purpose. It's the more
interesting path: the survivor (rank 0) doesn't crash, it blocks in a
collective waiting for a peer that's never coming back, and you can watch the
controller tear it down (`terminating generation 1 (survivors still up: [0])`).

Checkpoints land at steps 3 and 6 and the failure hits step 7, so the job has
both a real checkpoint to return to *and* real work to replay. A failure
before the first checkpoint would just restart from zero and demonstrate
nothing.

## Recording it as a GIF

No recording tool ships with this repo, and none is required to run the demo.
To produce `docs/demo.gif`:

```bash
brew install asciinema agg          # or: pipx install asciinema
asciinema rec demo/demo.cast -c "bash demo/demo.sh" --overwrite
agg demo/demo.cast docs/demo.gif --cols 100 --rows 32 --font-size 16
```

`asciinema` needs a real TTY, so run it from an interactive terminal — it will
not work from a non-interactive shell or CI runner.

Keep `demo.cast` out of the commit and the GIF in `docs/`; the `.cast` is
regenerable and the GIF is the artefact people actually look at. Both are
gitignored except `docs/demo.gif`.

Sizing notes, so the GIF stays legible when GitHub scales it down:

| setting | value | why |
|---|---|---|
| `--cols 100` | 100 columns | the widest line the controller prints is ~94 chars; narrower and the recovery table wraps |
| `--rows 32` | 32 rows | the whole run fits without scrolling, so no information scrolls past |
| `--font-size 16` | 16px | readable at GitHub's inline width (~880px) |

Trim the idle gaps rather than speeding the whole thing up — `agg
--speed` compresses the recovery pause too, and that pause is the point.
`asciinema rec --idle-time-limit 1.5` caps dead air while leaving real timings
intact.

## Where it goes in the README

Embed it immediately under the **Status** heading, before *Architecture* —
it's the fastest way for a reader to see that the thing works before they
commit to reading anything:

```markdown
![crash → detect → recover → resume](docs/demo.gif)
```

The main README already links this transcript from that spot, so dropping the
GIF in is a one-line change with nothing else to rewire.
