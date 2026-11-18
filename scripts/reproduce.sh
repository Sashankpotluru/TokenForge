#!/bin/sh
# Run from any directory with the locked benchmark environment active.
set -eu
cd "$(dirname "$0")/.."
python -m benchmarks.download "$@"
python -m benchmarks.measure
python -m benchmarks.evidence
