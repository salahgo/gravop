"""The public release trees must stay clean and in step with the working tree.

Two properties are checked, both of which have failed silently before in
projects of this shape: a release that carries development-only references or
machine paths, and a release that has drifted from the code it claims to
publish.
"""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
SYNC = os.path.join(ROOT, 'scripts', 'sync_public_artifacts.py')

# These check the release BUILDER, a maintainer tool that is deliberately not
# part of the release: its source spells out the patterns its own audit
# forbids. A reader running the suite from the published tree therefore has
# neither the builder nor public-artifacts/, and these tests must skip rather
# than fail. Every other test in the suite runs on the published tree alone;
# that is asserted by tests/test_release_completeness.py.
builder_absent = pytest.mark.skipif(
    not os.path.exists(SYNC),
    reason='release builder is not part of the release; nothing to audit here')


@builder_absent
@pytest.mark.verification
def test_public_tree_audit_is_clean():
    r = subprocess.run([sys.executable, SYNC, '--check'],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


@builder_absent
@pytest.mark.verification
def test_release_metadata_is_complete():
    import json
    meta = json.load(open(os.path.join(ROOT, 'release.json')))
    for key in ('version', 'title', 'description', 'creators', 'keywords',
                'repository'):
        assert meta.get(key), f'release.json is missing {key}'
    assert meta['creators'] and meta['creators'][0].get('affiliation')


@builder_absent
@pytest.mark.verification
def test_github_tree_carries_its_metadata():
    gh = os.path.join(ROOT, 'public-artifacts', 'github')
    for fn in ('LICENSE', 'README.md', 'CITATION.cff', '.zenodo.json',
               'MANIFEST.sha256', 'pyproject.toml'):
        assert os.path.exists(os.path.join(gh, fn)), fn


@builder_absent
@pytest.mark.verification
def test_github_tree_carries_its_community_and_publication_files():
    """A public repository is judged on these before anyone reads the code."""
    gh = os.path.join(ROOT, 'public-artifacts', 'github')
    for fn in ('CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'CHANGELOG.md',
               'DATA.md', 'REPRODUCING.md', 'LICENSES.md', 'setup-env.sh',
               'scripts/verify_release.sh', '.gitignore',
               '.github/PULL_REQUEST_TEMPLATE.md',
               '.github/workflows/verify.yml',
               '.github/ISSUE_TEMPLATE/bug_report.yml',
               '.github/ISSUE_TEMPLATE/reproduction_report.yml'):
        assert os.path.exists(os.path.join(gh, fn)), fn


@builder_absent
@pytest.mark.verification
def test_the_citation_file_is_valid_without_the_identifiers_that_do_not_exist():
    """CITATION.cff must parse and satisfy its schema at every stage.

    The CFF schema validates `doi` and `orcid` against patterns, so an
    unresolved placeholder in either makes the file invalid and breaks the
    citation widget. The builder therefore emits those keys only once the
    value is real. This gate holds both halves of that: the file is complete
    and parses now, and it carries no angle-bracket placeholder.
    """
    import re
    yaml = pytest.importorskip('yaml')
    gh = os.path.join(ROOT, 'public-artifacts', 'github')
    text = open(os.path.join(gh, 'CITATION.cff'), encoding='utf-8').read()
    cff = yaml.safe_load(text)
    assert cff['cff-version'] == '1.2.0'
    for key in ('type', 'title', 'abstract', 'version', 'license',
                'date-released', 'repository-code', 'authors', 'keywords'):
        assert cff.get(key), f'CITATION.cff is missing {key}'
    assert not re.search(r'<[A-Z_]+>', text), (
        'CITATION.cff carries an unresolved placeholder; the schema rejects '
        'one in doi or orcid, so those keys are emitted only when real')
    if 'doi' in cff:
        assert cff['doi'].startswith('10.'), cff['doi']
    for author in cff['authors']:
        if 'orcid' in author:
            assert author['orcid'].startswith('https://orcid.org/')


@builder_absent
@pytest.mark.verification
def test_the_deposition_metadata_is_what_zenodo_accepts():
    """Each tree's `.zenodo.json` describes its own record and holds no placeholder.

    The source release's file is read by Zenodo when a GitHub release is
    archived, and a malformed ORCID or DOI makes that archiving fail, so an
    unresolved value must be absent rather than visible. The archive's file
    is the draft of the companion dataset record and names the repository.
    """
    import json
    out = os.path.join(ROOT, 'public-artifacts')
    for tree, kind in (('github', 'software'), ('zenodo', 'dataset')):
        path = os.path.join(out, tree, '.zenodo.json')
        text = open(path, encoding='utf-8').read()
        meta = json.loads(text)
        for key in ('title', 'upload_type', 'description', 'creators',
                    'license', 'version', 'keywords'):
            assert meta.get(key), f'{tree}/.zenodo.json is missing {key}'
        assert meta['upload_type'] == kind, (tree, meta['upload_type'])
        assert not re.search(r'<[A-Z_]+>', text), (
            f'{tree}/.zenodo.json carries an unresolved placeholder')
        for c in meta['creators']:
            if 'orcid' in c:
                assert re.fullmatch(r'(https://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]',
                                    c['orcid']), c['orcid']
        if kind == 'dataset':
            assert any(r['relation'] == 'isSupplementTo' and r['scheme'] == 'url'
                       for r in meta.get('related_identifiers', [])), (
                'the archival record does not name the repository')


@builder_absent
@pytest.mark.verification
def test_the_publication_guide_names_every_outstanding_placeholder():
    """A placeholder nobody wrote down is a placeholder that ships."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import sync_public_artifacts as sync
    guide = open(os.path.join(ROOT, 'PUBLISHING.md'), encoding='utf-8').read()
    names = {'concept_doi': '<ZENODO_CONCEPT_DOI>',
             'version_doi': '<ZENODO_DOI>',
             'archive_doi': '<ARCHIVE_DOI>',
             'license_docs': '<LICENSE>',
             'creators[0].orcid': '<ORCID>',
             'paper.doi': '<PAPER_DOI>'}
    for field in sync.unresolved():
        assert names[field] in guide, (
            f'{field} is unresolved and PUBLISHING.md does not say where '
            'its value comes from')


@builder_absent
@pytest.mark.verification
def test_zenodo_tree_carries_the_result_records():
    zen = os.path.join(ROOT, 'public-artifacts', 'zenodo')
    n = sum(len([f for f in fs if f.endswith('.json')])
            for _, _, fs in os.walk(zen))
    assert n > 500, f'only {n} result records in the archive tree'
    for fn in ('MANIFEST.sha256', 'ARCHIVE.md', '.zenodo.json'):
        assert os.path.exists(os.path.join(zen, fn)), fn


@builder_absent
@pytest.mark.verification
def test_the_archive_does_not_duplicate_the_source_release():
    """Zenodo holds what GitHub does not, and nothing else.

    `.zenodo.json` and `MANIFEST.sha256` are the only names both trees
    carry, and each describes its own tree.
    """
    out = os.path.join(ROOT, 'public-artifacts')
    # SYNC_REPORT.md is the maintainer's audit of the trees against the
    # manuscript; it sits beside them and is in neither.
    assert sorted(os.listdir(out)) == ['SYNC_REPORT.md', 'github', 'zenodo'] \
        or sorted(os.listdir(out)) == ['github', 'zenodo']
    zen = os.path.join(out, 'zenodo')
    # results/: every result record and the figures; experiments/: the
    # closure-test and uncertainty records and measurements; data/: the
    # baryonic dataset the uncertainty studies read.
    assert sorted(os.listdir(zen)) == ['.zenodo.json', 'ARCHIVE.md',
                                       'MANIFEST.sha256', 'data', 'experiments',
                                       'results']
    gh = os.path.join(out, 'github')
    shared = []
    for dirpath, _, filenames in os.walk(zen):
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), zen)
            if os.path.exists(os.path.join(gh, rel)):
                shared.append(rel)
    assert sorted(shared) == ['.zenodo.json', 'MANIFEST.sha256'], shared
