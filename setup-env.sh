#!/usr/bin/env bash
# Create a virtual environment, install the package, and run the gate suite.
# This is the documented install path and the one a third party should use.
# `requirements.lock` is a record of the environment that produced the
# published numbers, not an install script: its torch line is a local file
# reference and does not resolve elsewhere. See the header of that file.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[all]"

python - <<'PY'
import gravop, statecond
print('gravop and statecond import cleanly')
PY

if [ -d results ]; then
  python -m pytest tests/ -q
else
  cat <<'MSG'

The archival record is not unpacked here, so the gates that read stored
results cannot run. Extract the deposit's results/ at the root of this
checkout (unzip <record>.zip 'results/*') and run:

    python -m pytest tests/ -q

Running now without it, to check what the source tree alone can establish:
MSG
  python -m pytest tests/ -q -m verification || true
fi
