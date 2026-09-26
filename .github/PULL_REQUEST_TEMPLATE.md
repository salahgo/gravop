## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- What was wrong, or what is now possible. -->

## Evidence

<!-- What you ran and what passed. Paste the counts. -->

```
python -m pytest tests/ -q
```

## Checklist

- [ ] The gate suite passes, or the failures are explained above.
- [ ] Any new gate declares `verification` or `regression`.
- [ ] No reported result moved, or the gate that pins it was updated with the
      record the new value comes from.
- [ ] No generated data, trained weights or result artefacts are committed.
- [ ] `CHANGELOG.md` has an entry if the change is visible
      to a user.
