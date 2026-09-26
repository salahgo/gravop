#!/usr/bin/env bash
# R5 -- the from-scratch reproduction path, measured rather than assumed.
#
# WHY THIS EXISTS
# ---------------
# Everything in this project reproduces on the machine it was developed on,
# which is exactly the claim a referee cannot check. This script builds the
# path a stranger takes -- clean clone, fresh venv, install, regenerate the
# cheap data, run the gates -- and records what it costs and what breaks.
#
# It clones from the local git repository rather than copying the working
# tree, so anything gitignored is ABSENT, which is the point: data/d1,
# data/sc, data/d3_fischer2025 and results/statecond/ckpt are all absent on a
# fresh clone and the failures they cause are part of the finding.
#
# The known landmine is the Triton backward trap: where the Python development
# headers are missing and cannot be installed, any op whose BACKWARD is
# routed to a Triton kernel fails at runtime while every forward-only smoke
# test passes. This script must be what catches that, not a stranger.
#
# Usage:  bash scripts/clean_env_smoke.sh [workdir]
# Output: $workdir/report.txt, and a summary on stdout.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-/tmp/sc_clean_$(date +%Y%m%d_%H%M%S)}"
REPORT="$WORK/report.txt"
mkdir -p "$WORK"
: > "$REPORT"

FAILURES=0

step() {                     # step <label> <command...>
  local label="$1"; shift
  local t0 rc
  t0=$(date +%s)
  echo "=== $label" | tee -a "$REPORT"
  "$@" >>"$REPORT" 2>&1
  rc=$?
  local dt
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -ne 0 ]; then FAILURES=$((FAILURES + 1)); fi
  printf '%-44s rc=%-3s %8.1f s\n' "$label" "$rc" "$dt" | tee -a "$WORK/summary.txt"
  return 0
}

echo "clean-environment smoke test"          | tee -a "$REPORT"
echo "source repo : $SRC"                    | tee -a "$REPORT"
echo "workdir     : $WORK"                   | tee -a "$REPORT"
echo "started     : $(date -Is)"             | tee -a "$REPORT"
: > "$WORK/summary.txt"

step "git clone (gitignored files absent)" \
     git clone --quiet "$SRC" "$WORK/repo"

step "python -m venv" \
     python3 -m venv "$WORK/venv"

PY="$WORK/venv/bin/python"
step "pip install -e .[all]" \
     "$WORK/venv/bin/pip" install --quiet -e "$WORK/repo[all]"

cd "$WORK/repo" || exit 1

step "import the package" \
     "$PY" -c "import statecond, gravop; print(statecond.__file__)"

step "regenerate data/sc (four bespoke systems)" \
     "$PY" scripts/sc_gen.py --all --n 600

# sidm is imported from data/d1, which is NOT in git. This is expected to fail
# on a clean clone and the message it gives is part of what R5 measures.
step "missing SIDM input is diagnosed" \
     "$PY" -c 'import subprocess,sys; r=subprocess.run([sys.executable,"scripts/sc_import_sidm.py"],capture_output=True,text=True); assert r.returncode != 0 and "data/d1" in r.stdout+r.stderr and "Traceback" not in r.stderr, r.stdout+r.stderr'

step "artefact-only gates (no data, no checkpoints)" \
     "$PY" -m pytest tests/ -q -m regression

step "verification gates" \
     "$PY" -m pytest tests/ -q -m verification

# The Triton backward trap: a forward pass goes through cuBLAS/cuFFT and
# passes, a backward pass through a Triton kernel does not. Train one step of
# every backbone, on both n_fields = 1 and 2.
step "backward pass on every backbone (the Triton trap)" \
     "$PY" - <<'PYEOF'
import torch
from statecond import BACKBONES
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for name, cls in BACKBONES.items():
    for n_fields in (1, 2):
        net = cls(n_fields=n_fields, n_par=4, cond_extra=1, n_x=128).to(dev)
        u = torch.randn(4, n_fields, 128, device=dev)
        c = torch.randn(4, 5, device=dev)
        net(u, c).square().mean().backward()
        print(f'{name} n_fields={n_fields} backward OK', flush=True)
PYEOF

echo "finished    : $(date -Is)" | tee -a "$REPORT"
echo
echo "--- summary ---"
cat "$WORK/summary.txt"
echo
echo "full log: $REPORT"

if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES required steps failed" >&2
  exit 1
fi
