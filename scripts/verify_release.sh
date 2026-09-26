#!/usr/bin/env bash
# Verify one or both published trees: every file against its manifest, and the
# digest of that manifest, which is what names the snapshot.
#
# Usage:  scripts/verify_release.sh <tree> [<tree> ...]
#
# Compare each digest with the one quoted for the release you expect. No
# document inside a tree states that tree's own digest: writing it in would
# change it. The source release is also named by its version tag.
set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <tree> [<tree> ...]" >&2
  exit 2
fi

status=0
for tree in "$@"; do
  if [ ! -f "$tree/MANIFEST.sha256" ]; then
    echo "$tree: no MANIFEST.sha256" >&2
    status=1
    continue
  fi
  n=$(grep -c . "$tree/MANIFEST.sha256")
  if (cd "$tree" && sha256sum --quiet -c MANIFEST.sha256); then
    digest=$(sha256sum "$tree/MANIFEST.sha256" | cut -d' ' -f1)
    echo "$tree: $n files verified   tree digest ${digest:0:16}  ($digest)"
  else
    echo "$tree: FAILED -- the tree does not match its manifest" >&2
    status=1
  fi
done
exit "$status"
