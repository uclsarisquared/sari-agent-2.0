#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$script_dir/logs/$run_stamp"
mkdir -p "$log_dir"

policies=(a5 a7-no-stop-guard replan-test)
statuses=()
failed=0

trap 'exit 130' INT TERM

for policy in "${policies[@]}"; do
    config="$script_dir/$policy.toml"
    log="$log_dir/$policy.log"
    status_file="$log_dir/$policy.status"
    printf 'Running %-18s -> %s\n' "$policy" "$log"
    (
        cd "$repo_root"
        uv run python -m sari_bench run --config "$config" --time-limit 80 --tries 2
    ) >"$log" 2>&1
    status=$?
    printf '%s\n' "$status" >"$status_file"
    statuses+=("$status")
    if (( status != 0 )); then
        failed=1
    fi
done

printf '\n%-18s %s\n' "CONFIGURATION" "STATUS"
printf '%-18s %s\n' "------------------" "------"
for index in "${!policies[@]}"; do
    status=${statuses[$index]}
    label="ok"
    if (( status != 0 )); then
        label="failed ($status)"
    fi
    printf '%-18s %s\n' "${policies[$index]}" "$label"
done
printf '\nLogs and status files: %s\n' "$log_dir"

exit "$failed"