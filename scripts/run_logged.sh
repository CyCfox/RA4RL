#!/usr/bin/env bash
# Run inside the project container, retaining exit status after --rm removes it.
set -euo pipefail
test -f /.dockerenv || { echo 'Run inside Docker only' >&2; exit 1; }
task_launch_dir="$1"
shift
test -d "$task_launch_dir"
test ! -e "$task_launch_dir/status.json"
task_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
task_child_pid=
task_stop_signal=
task_record_exit() {
  task_exit_code=$?
  task_phase=completed
  if [[ "$task_exit_code" != 0 ]]; then task_phase=failed; fi
  if [[ -n "$task_stop_signal" ]]; then task_phase=interrupted; fi
  printf '{"phase":"%s","exit_code":%d,"started_at":"%s","finished_at":"%s"}\n' \
    "$task_phase" "$task_exit_code" "$task_started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$task_launch_dir/.status.tmp"
  mv "$task_launch_dir/.status.tmp" "$task_launch_dir/status.json"
}
trap task_record_exit EXIT
task_forward_stop() {
  task_stop_signal="$1"
  if [[ -n "$task_child_pid" ]]; then
    kill -TERM "$task_child_pid" 2>/dev/null || true
    wait "$task_child_pid" 2>/dev/null || true
  fi
  if [[ "$task_stop_signal" == INT ]]; then exit 130; fi
  exit 143
}
trap 'task_forward_stop TERM' TERM
trap 'task_forward_stop INT' INT
printf '{"phase":"running","started_at":"%s"}\n' "$task_started_at" > "$task_launch_dir/status.json"
"$@" > "$task_launch_dir/run.log" 2>&1 &
task_child_pid=$!
wait "$task_child_pid"
