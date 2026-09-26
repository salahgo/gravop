#!/usr/bin/env bash
# Run the full E22 protocol for one backbone over all five systems.
#
# `sc_experiment.py --all` iterates statecond.pdes.SYSTEMS, which holds the four
# bespoke PDEs only -- `sidm` enters through sc_import_sidm.py and so has to be
# named explicitly. Forgetting that produces a four-system result that looks
# complete, so the two calls live here together.
#
# Usage: scripts/sc_run_backbone.sh <backbone> [device] [epochs]
set -euo pipefail

BACKBONE="${1:?usage: sc_run_backbone.sh <backbone> [device] [epochs]}"
DEVICE="${2:-cuda:0}"
EPOCHS="${3:-200}"
cd "$(dirname "$0")/.."

for target in "--all" "--system sidm"; do
    # shellcheck disable=SC2086
    python scripts/sc_experiment.py $target \
        --backbone "$BACKBONE" --device "$DEVICE" --epochs "$EPOCHS" \
        --ckpt-dir
done

python scripts/sc_law.py
