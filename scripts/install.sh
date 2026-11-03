#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv || { echo 'Run inside Docker only' >&2; exit 1; }
# Keep the image's CUDA-enabled torch; every installation is project-local.
python -m venv --system-site-packages .runtime/venv
task_requirements=requirements.txt
if test -f docker/requirements.lock.txt; then
  task_requirements=docker/requirements.lock.txt
fi
.runtime/venv/bin/python -m pip install -r "$task_requirements" \
  --constraint docker/constraints.txt --cache-dir .runtime/cache
.runtime/venv/bin/python -m pip freeze > .runtime/requirements.resolved.txt
.runtime/venv/bin/python -m pip freeze --local > .runtime/requirements.local.txt
