"""Provenance stamps, so a stale artefact announces itself.

WHY THIS EXISTS
---------------
Twice now an experiment has landed after another experiment it invalidated, and
the second was never recomputed:

  * P13 (`contract.json`) compared arm C against arm D. P20 then made arm B
    through a converged clock the stronger comparator in three systems, and
    P13 was not re-run for a day and a half of project time -- long enough for
    both numbers to be quoted side by side in the same document.
  * P16 (`reference_accuracy.json`) hard-coded the time head's error at
    1.8-7.1%. P20 improved the head by 6-110x, which turned two of the five
    margins from "comfortable" to "below one".

Neither was a reasoning failure; both were bookkeeping failures, and bookkeeping
failures are what gates are for. The rule, stated explicitly:
an experiment that lands after another it could invalidate must recompute that
artefact or record why it cannot.

WHY NOT FILE MTIMES
-------------------
`results/statecond/*.json` is committed, so on a fresh clone every artefact has
the same checkout mtime and the ordering information is gone. The stamp has to
live *inside* the artefact. Each producer records when it ran and, for each
input artefact it consumed, the stamp that input carried at the time. The gate
then compares recorded-input-stamp against current-producer-stamp: if a producer
has been regenerated since its consumer read it, the two disagree and the
consumer is stale.

USAGE
-----
    from sc_provenance import stamp, read_stamp
    rec['_provenance'] = stamp('sc_contract.py',
                               inputs={'direct.json': ..., 'timehead.json': ...})

`inputs` maps artefact filename -> the loaded JSON (or None if absent). The
helper extracts each one's own stamp; a missing or unstamped input is recorded
as None, which the gate reports as "unknown" rather than failing, so artefacts
predating this mechanism do not break the suite.
"""
import datetime
import os

RES = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond')


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds')


def read_stamp(obj):
    """The `generated` field of a loaded artefact, or None.

    Handles both layouts in use here: a top-level `_provenance` (timehead,
    contract, direct write one record per system under the system key, plus a
    top-level block) and a per-system one.
    """
    if not isinstance(obj, dict):
        return None
    p = obj.get('_provenance')
    if isinstance(p, dict):
        return p.get('generated')
    for v in obj.values():                     # per-system layout
        if isinstance(v, dict) and isinstance(v.get('_provenance'), dict):
            return v['_provenance'].get('generated')
    return None


def stamp(script, inputs=None):
    """Build a provenance record for the artefact this script is writing."""
    return {
        'script': script,
        'generated': _now(),
        'inputs': {k: read_stamp(v) for k, v in (inputs or {}).items()},
    }


# The dependency graph, declared once. consumer -> the artefacts it reads and
# whose change should invalidate it. Only edges that have actually bitten, plus
# the ones structurally identical to them; a graph nobody maintains is worse
# than none.
EDGES = {
    'contract.json': ['direct.json', 'timehead.json'],
    'reference_accuracy.json': ['timehead.json'],
    'timehead.json': [],
    'direct.json': [],
}


def check(res_dir=RES, strict=False):
    """Returns (ok, list of human-readable problems)."""
    import json
    problems = []
    loaded = {}
    for name in set(EDGES) | {x for v in EDGES.values() for x in v}:
        p = os.path.join(res_dir, name)
        loaded[name] = json.load(open(p)) if os.path.exists(p) else None

    for consumer, deps in EDGES.items():
        c = loaded.get(consumer)
        if c is None:
            if strict: problems.append(f'{consumer} is missing')
            continue
        cs = read_stamp(c)
        if cs is None:
            if strict: problems.append(f'{consumer} has no provenance stamp')
            continue                       # predates the mechanism; not a fail
        prov = c.get('_provenance') or next(
            (v['_provenance'] for v in c.values()
             if isinstance(v, dict) and isinstance(v.get('_provenance'), dict)),
            {})
        recorded = prov.get('inputs', {})
        for dep in deps:
            d = loaded.get(dep)
            if d is None:
                if strict: problems.append(f'{consumer}: required dependency {dep} is missing')
                continue
            current = read_stamp(d)
            if current is None:
                if strict: problems.append(f'{consumer}: dependency {dep} has no provenance stamp')
                continue                   # dependency not yet instrumented
            was = recorded.get(dep)
            if was is None:
                problems.append(
                    f'{consumer} does not record which {dep} it used; '
                    f'regenerate it so the edge is auditable')
            elif was != current:
                problems.append(
                    f'{consumer} is STALE: it was computed against {dep} '
                    f'stamped {was}, but {dep} now reads {current}. '
                    f'Re-run the script that writes {consumer}.')
    return (not problems), problems


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true', help='allow absent historical artifacts/stamps')
    parser.add_argument('--results', default=RES)
    args = parser.parse_args()
    ok, probs = check(args.results, strict=not args.legacy)
    for p in probs:
        print('STALE:', p)
    print('provenance:', 'OK' if ok else f'{len(probs)} problem(s)')

    sys.exit(0 if ok else 1)
