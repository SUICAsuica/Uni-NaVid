#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TEST_CASE="${1:-test_cases/vln_1}"
OUTPUT_DIR="${2:-outputs/$(basename "$TEST_CASE")}"

mkdir -p "$OUTPUT_DIR"
exec .venv/bin/python offline_eval_uninavid.py "$TEST_CASE" "$OUTPUT_DIR"
