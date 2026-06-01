#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec .venv/bin/python tools/uninavid_server.py "$@"
