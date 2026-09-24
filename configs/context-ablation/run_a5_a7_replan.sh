#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/_common.sh"
log_dir="$script_dir/logs/$run_stamp-comprehensive"
prompts="$log_dir/comprehensive_prompts.json"
mkdir -p "$log_dir"

policies=(a5 a7-no-stop-guard replan-test)

build_prompts "$prompts" easy_prompts.json medium_prompts.json hard_prompts.json

for policy in "${policies[@]}"; do
    run_policy Running "$policy" --prompts "$prompts" --queue-mode prompt-first \
        --time-limit 80 --tries 3 --name "context-ablation-$policy-comprehensive"
done

print_summary "Prompt manifest, logs, and status files"
