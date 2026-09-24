#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/_common.sh"
log_dir="$script_dir/logs/$run_stamp-comprehensive"
prompts="$log_dir/comprehensive_prompts.json"
mkdir -p "$log_dir"

# baseline and a1 already finished in a prior comprehensive run; this picks up where that left off.
# hard-baseline is intentionally omitted: it differs from baseline only by using the hard-only
# prompt battery, so overriding it with the comprehensive battery would duplicate baseline.
policies=(a2c a3 a4 a5 a6-2 a6-4 a7-no-stop-guard)

build_prompts "$prompts" easy_prompts.json medium_prompts.json hard_prompts.json

for policy in "${policies[@]}"; do
    run_policy Running "$policy" --prompts "$prompts" --tries 3 \
        --name "context-ablation-$policy-comprehensive"
done

print_summary "Prompt manifest, logs, and status files"
