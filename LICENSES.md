# Licensing

This release contains four kinds of material with three different answers.
Each is stated here rather than left to be inferred from the presence of a
single `LICENSE` file.

## Code — GPL-3.0-or-later

Everything under `src/`, `scripts/` and `tests/` is licensed under the GNU
General Public License, version 3 or later. The full text is in
[`LICENSE`](LICENSE).

This is a requirement rather than a preference. The gravothermal solver
follows the algorithmic lineage of the GPLv3 GravothermalSIDM implementation
against which it is cross-validated, so the copyleft term is inherited. A
permissive licence is not available for this component.

## Documentation and figures — `<LICENSE>`

The Markdown documents and the rendered figures in the archival record are
**not yet assigned a licence**. This is an open decision,
marked here rather than silently defaulted.

Recommended: CC-BY-4.0, which is the usual choice for research documentation
and figures, is compatible with GPL-3.0-or-later code in the same deposit, and
permits reuse with attribution.

Constraint to check before deciding: a figure that is also published under a
publisher's agreement is governed by that agreement as well. The licence is
recorded here and in the deposition metadata once that is settled.

## Machine-readable results — same terms as the code

The archival record holds `results/`: the numerical output of the released
code. It carries the licence recorded in the deposition metadata,
`GPL-3.0-or-later`, so that a reader who takes a result and the code that
produced it is under one set of terms.

## External data — not redistributed

Three external datasets and one external code are used; none is included in
either tree.

| Source | Condition |
|---|---|
| Public isolated-halo N-body core-collapse run of Fischer, Yu & Dolag (2025) | Obtained from the authors' public distribution. Cite the paper if you use it. Not redistributed here. |
| SIDM Concerto public data release (Zenodo, CC-BY) | Cite the SIDM Concerto paper and its Zenodo release. Not redistributed here. |
| SIDM Concerto merger trees for one host, supplied by the suite's authors on request | Cited to the SIDM Concerto release, of which they are intended to be part. Not redistributed here; request from the suite's authors. |
| Public GravothermalSIDM reference implementation | GPLv3. Cloned separately for cross-validation; not vendored here. |

[`DATA.md`](DATA.md) records each source, its URL where one exists, its
access conditions and how it was verified.

## Third-party dependencies

Dependencies are declared in [`pyproject.toml`](pyproject.toml) and pinned in
[`requirements.lock`](requirements.lock). They keep their own licences; none
is vendored into this repository.

## Citing

See the *Citation* section of [`README.md`](README.md) and
[`CITATION.cff`](CITATION.cff).
