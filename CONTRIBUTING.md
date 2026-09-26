# Contributing

Contributions are welcome: bug reports, reproduction reports, portability
fixes, documentation, and new benchmark systems.

## Before you start

Open an issue first for anything that changes a published number, a protocol,
or the acceptance suite. Those changes need a discussion about evidence, not
just about code.

Small fixes — typos, broken links, portability patches, clearer error
messages — can go straight to a pull request.

## Development setup

```bash
git clone https://github.com/salahgo/gravop
cd gravop
./setup-env.sh          # virtual environment, pinned install, gate suite
```

The acceptance suite needs the archival record's `results/` extracted at the
root of the checkout. Without it, the tests that read stored results fail
rather than skip; see [`REPRODUCING.md`](REPRODUCING.md) for how the two trees
fit together.

## The rule this project runs on

The acceptance suite checks the reported results against the stored records or
recomputes them from a source of truth. Each gate declares which kind it is:

```bash
python -m pytest tests/ -q                  # all gates
python -m pytest tests/ -q -m verification  # recomputed from a source of truth
python -m pytest tests/ -q -m regression    # compared with a stored artefact
```

A `verification` gate computes its assertion in the session from the solver,
the data, the model code or a checkpoint's own behaviour, so a wrong number
stored under `results/` cannot make it pass. A `regression` gate pins internal
consistency and provenance. New gates must carry one marker or the other.

If a change moves a reported result, update the gate that pins it together
with the record it reads, and say in the pull request which result moved and
why.

## Pull requests

- One topic per pull request.
- Match the surrounding style; the code is plain NumPy, SciPy, Numba and
  PyTorch with no framework layer.
- Add or update a gate for any behaviour you change.
- Say in the description what you ran and what passed. "The suite passes" with
  the count is enough.
- Do not commit generated data, trained weights or result artefacts. They are
  ignored by `.gitignore` and belong in the archival deposit.

## Adding a benchmark system

A new system needs: the equation and its event definition, a generator with a
recorded seed, an entry in the specification table, and convergence evidence
that the event time is resolved far better than the effects being measured.
[`STATECOND_BENCHMARK.md`](STATECOND_BENCHMARK.md) documents the schema every
system follows.

## Reporting a reproduction problem

Use the reproduction report issue template. Include the snapshot digests you
are working from, your platform and Python version, and the exact command and
output. Digest mismatches are the single most useful thing to report, because
they say whether the trees or the environment moved.

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the licence of the file they touch: code
under GPL-3.0-or-later, documentation and figures under the terms recorded in
[`LICENSES.md`](LICENSES.md).
