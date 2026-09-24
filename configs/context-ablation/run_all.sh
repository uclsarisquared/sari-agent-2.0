#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/_common.sh"
log_dir="$script_dir/logs/$run_stamp"
mkdir -p "$log_dir"

policies=(baseline a1 a2c a3 a4 a5 a6-2 a6-4 a7-no-stop-guard hard-baseline)

for policy in "${policies[@]}"; do
    run_policy Running "$policy"
done

print_summary "Logs and status files"
