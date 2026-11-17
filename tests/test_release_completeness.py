"""A published document may not name a file the published trees do not hold.

This gate exists because an earlier review bundle failed it. Six rows of its
review-stage checklist -- the dataset split manifest, the per-trajectory
coverage records, the gate decisions, the per-halo catalogue with the
cosmological IDs, the experiment ledger, and the external-data access
conditions -- were marked *present* and pointed at paths that were in neither
the source tree nor the archive record; and the manuscript, which the release's
headline check reads every number out of, was in neither tree either. The archive builder copied `.json` only, and the source tree
shipped a fixed list of root files that had not been revised in months, so the
claims and the trees drifted apart silently. No regression gate could see it:
every stored number stayed consistent with every other stored number.

The check is a reachability one. It parses the documents a reviewer is handed,
extracts every repo-relative path they name in backticks, and requires each to
resolve inside one of the two trees. Exemptions are listed one by one with the
reason, because an unexplained exemption is how this failed the first time.
"""
import json
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GH = os.path.join(ROOT, 'public-artifacts', 'github')
ZEN = os.path.join(ROOT, 'public-artifacts', 'zenodo')

needs_release = pytest.mark.skipif(
    not os.path.isdir(GH),
    reason='release trees are a build product and are not part of the release')

# Documents a reviewer is handed, and which tree they are handed them in.
DOCUMENTS = ['README.md', 'REPRODUCING.md', 'DATA.md', 'CONTRIBUTING.md',
             'LICENSES.md', 'CHANGELOG.md', 'STATECOND_BENCHMARK.md']

# Paths a published document names on purpose without shipping them.
EXEMPT = {
    # The release builder is a maintainer tool, not part of the release.
    'scripts/sync_public_artifacts.py',
    # The five HDF5 benchmark sets are regenerable from the released
    # generators with the recorded seeds; their generation record ships as
    # data/sc/manifest.json and their schema is in the readers.
    'data/sc/',
    'data/sc/rd.h5', 'data/sc/ks.h5', 'data/sc/film.h5',
    'data/sc/burgers.h5', 'data/sc/sidm.h5',
    # The application training set, likewise regenerated (DATA.md).
    'data/d1/',
    # Historical application rebuild, named in README as historical.
    'scripts/run_all.sh',
}

# The archive record is unpacked into the checkout root by a reviewer, so a
# results/ path counts as reachable if it is in either tree.
PATH = re.compile(
    r'`((?:src|tests|scripts|results|data|docs|paper)/[A-Za-z0-9_./-]*'
    r'|[A-Z][A-Za-z0-9_]*\.(?:md|json|lock|txt|sh|cff|bib)'
    r'|pyproject\.toml|requirements\.(?:txt|lock)|setup-env\.sh)`')


def _reachable(rel):
    return (os.path.exists(os.path.join(GH, rel))
            or os.path.exists(os.path.join(ZEN, rel)))


def _named_paths(text):
    out = set()
    for m in PATH.finditer(text):
        rel = m.group(1)
        if '*' in rel or rel in EXEMPT:
            continue
        if rel.startswith('public-artifacts/'):
            continue        # names the trees themselves
        out.add(rel)
    return out


@needs_release
@pytest.mark.verification
def test_every_path_a_published_document_names_is_in_the_release():
    dangling = {}
    for doc in DOCUMENTS:
        path = os.path.join(GH, doc)
        assert os.path.exists(path), f'{doc} is not in the source release'
        missing = sorted(r for r in _named_paths(open(path, encoding='utf-8').read())
                         if not _reachable(r))
        if missing:
            dangling[doc] = missing
    assert not dangling, (
        'published documents name paths no published tree holds:\n' +
        '\n'.join(f'  {d}: {", ".join(m)}' for d, m in dangling.items()))


@needs_release
@pytest.mark.verification
def test_the_verification_suite_has_its_inputs_in_the_release():
    """Every input the shipped suite reads, other than the ones it skips on.

    Determined empirically: the suite was run against a checkout assembled
    from the two trees alone, and each of these was a failure until it
    shipped.
    """
    for rel in ('STATECOND_BENCHMARK.md',
                'data/sc/manifest.json',
                'results/figures/fig20_work_precision.pdf',
                'results/e32/controls.npz',
                'results/statecond/conformal_deployed_records.npz'):
        assert _reachable(rel), f'the shipped test suite reads {rel}; no tree has it'


@needs_release
@pytest.mark.verification
def test_the_manifests_name_the_trees_they_ship_in():
    """Each manifest lists exactly the files of its tree, with their hashes.

    A manifest that does not match its tree is worse than none, because it
    invites the reviewer to conclude the tree was edited after the fact.
    Neither tree records its own digest or the other's, so each digest names
    its tree independently of the other.
    """
    import hashlib
    for tree in (GH, ZEN):
        rows = dict(reversed(line.split('  ', 1)) for line in
                    open(os.path.join(tree, 'MANIFEST.sha256')).read().splitlines())
        files = {os.path.relpath(os.path.join(d, f), tree)
                 for d, _, fs in os.walk(tree) for f in fs} - {'MANIFEST.sha256'}
        assert set(rows) == files, f'{tree}: manifest and tree list different files'
        for rel, h in rows.items():
            assert hashlib.sha256(open(os.path.join(tree, rel), 'rb').read()
                                  ).hexdigest() == h, f'{tree}: {rel} changed'
    assert not os.path.exists(os.path.join(ZEN, 'SNAPSHOT.json'))


@needs_release
@pytest.mark.verification
def test_no_document_inside_a_tree_quotes_that_tree_s_digest():
    """The staleness this forbids is structural, not accidental.

    A file inside a tree contributes to that tree's manifest, so writing the
    digest into it changes the digest, and the quoted value is wrong the
    instant it is written. The earlier bundle did exactly this and shipped two
    digests that named a tree which no longer existed. The archive's digest is
    therefore quoted only in the manuscript, which is outside the archive, and
    the source digest nowhere, which makes the failure impossible rather than
    merely detected.
    """
    # Only the documents that DESCRIBE the snapshot are constrained. Result
    # records carry content hashes because that is what they are for, and
    # DATA.md quotes checksums of externally supplied files, which are not
    # members of any tree here.
    sha = re.compile(r'\b[0-9a-f]{64}\b')
    for path in (os.path.join(GH, 'README.md'), os.path.join(GH, 'REPRODUCING.md'),
                 os.path.join(ZEN, 'ARCHIVE.md')):
        doc = os.path.relpath(path, ROOT)
        text = open(path, encoding='utf-8').read()
        assert not sha.search(text), (
            f'{doc} quotes a 64-hex digest; it is inside the tree whose digest '
            'that would be, so the value is wrong the instant it is written')


@pytest.mark.verification
def test_the_archive_digest_does_not_depend_on_the_clock():
    """The digest that names the archive must not change with the calendar.

    `CITATION.cff` lives inside the archive tree, so any field generated from
    `date.today()` rewrites that tree's manifest -- and therefore its digest --
    once a day, while nothing scientific has changed. The manuscript quotes
    that digest, so a referee recomputing it on a later date would find a value
    the paper does not contain. The release date is pinned in the tracked
    release record instead.

    This checks the property rather than a stored value: it regenerates the
    citation metadata for two different "today"s and requires the bytes to
    match.
    """
    import datetime
    import importlib.util

    # The builder is a maintainer tool and not part of the release, so on a
    # checkout assembled from the two trees this must skip, as the other
    # builder tests do. It
    # raised FileNotFoundError instead, and the review package refused to
    # write because of it: exactly the failure a referee would have seen.
    path = os.path.join(ROOT, 'scripts', 'sync_public_artifacts.py')
    if not os.path.exists(path):
        pytest.skip('release builder is not part of the release')
    spec = importlib.util.spec_from_file_location('sync_pa', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    meta = json.load(open(os.path.join(ROOT, 'release.json'), encoding='utf-8'))
    assert meta.get('date_released'), (
        'release.json carries no pinned date_released, so the citation file '
        'falls back to the clock and the archive digest drifts daily')

    real = datetime.date

    class Shifted(real):
        @classmethod
        def today(cls):
            return real(2099, 12, 31)

    first = mod.citation_cff(meta['version'], None)
    mod.datetime.date = Shifted
    try:
        second = mod.citation_cff(meta['version'], None)
    finally:
        mod.datetime.date = real
    assert first == second, (
        'the citation file changes with the system date, so the archive digest '
        'is not a stable name for the snapshot')
    assert meta['date_released'] in first


def _pyproject_version(root):
    text = open(os.path.join(root, 'pyproject.toml'), encoding='utf-8').read()
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


@pytest.mark.regression
def test_package_version_matches_release():
    """pyproject.toml carries the release version.

    The release version lives in release.json in the development tree and in
    CITATION.cff in the source release. `gravop.__version__` is not compared:
    src/gravop/__init__.py is hash-pinned by the population study's frozen
    configuration and cannot change until that study finishes.
    """
    rel = os.path.join(ROOT, 'release.json')
    if os.path.exists(rel):
        version = json.load(open(rel))['version']
    else:
        cff = open(os.path.join(ROOT, 'CITATION.cff'), encoding='utf-8').read()
        version = re.search(r'^version:\s*"([^"]+)"', cff, re.M).group(1)
    assert _pyproject_version(ROOT) == version
