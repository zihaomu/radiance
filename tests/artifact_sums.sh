#!/bin/sh
# artifact_sums.sh -- checks committed built artifacts against tests/ARTIFACTS.
#
# Some committed files are not sources but a build of sources: core/server/dashboard_page.h is a
# Node build of dashboard/, committed because the engine's build has no Node. The digest recorded
# INSIDE such an artifact proves only that the sources and the artifact's own claim agree --
# tampering with the artifact together with its embedded digest reads as an honest build. This
# manifest hashes the artifact itself from outside, the mechanical half that vendor/SHA256SUMS is
# to vendor/UPSTREAM: the dashboard_page test keeps sources and header in step, this one checks
# the served bytes. (A tamper that updates the artifact and this manifest in one breath remains,
# exactly as with a vendor file -- the manifest is against drift and lone tampering, not against
# a commit that rewrites the whole record.)
#
#   tests/artifact_sums.sh <repo-root>
#
# When an artifact is rebuilt for real, re-record it in the same commit:
#   sha256sum $(awk '{print $2}' tests/ARTIFACTS) > tests/ARTIFACTS
set -e
root=${1:?usage: artifact_sums.sh <repo-root>}
cd "$root"
command -v sha256sum >/dev/null || { echo "artifact_sums: sha256sum is required" >&2; exit 1; }
[ -f tests/ARTIFACTS ] || { echo "artifact_sums: tests/ARTIFACTS is missing" >&2; exit 1; }
sha256sum -c --status tests/ARTIFACTS || {
    echo "artifact_sums: a committed artifact differs from its recorded hash;" \
         "run sha256sum -c tests/ARTIFACTS to see which" >&2
    exit 1; }
echo "artifact_sums: $(wc -l < tests/ARTIFACTS) artifact(s), every hash matches"
