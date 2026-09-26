#!/usr/bin/env bash
# Train the two phase-conditioned production ensembles P2 needs.
#
# They differ only in the radius the phase is read at, which must be a radius
# the relevant observation actually resolves -- an ensemble trained at the
# emulator grid's innermost point (0.0032 r_s, index 0) cannot be conditioned by
# either application without extrapolating below the data:
#
#   --phi-index 13 (0.0091 r_s) -> E11; its N-body profiles resolve to 0.0084
#   --phi-index 23 (0.0207 r_s) -> E9;  its mock is sampled from 0.02 outward
#
# Usage: scripts/p2_ensembles.sh [device]
set -euo pipefail
DEVICE="${1:-cuda:0}"
cd "$(dirname "$0")/.."
source .venv/bin/activate

python scripts/e28_phase_ensemble.py --seeds 5 --epochs 200 \
    --device "$DEVICE" --phi-index 13 --suffix _nbody
python scripts/e28_phase_ensemble.py --seeds 5 --epochs 200 \
    --device "$DEVICE" --phi-index 23 --suffix _e9
echo "P2 ENSEMBLES COMPLETE"
