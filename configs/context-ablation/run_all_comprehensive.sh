#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/_common.sh"
log_dir="$script_dir/logs/$run_stamp-comprehensive"
prompts="$log_dir/comprehensive_prompts.json"
mkdir -p "$log_dir"

# hard-baseline is intentionally omitted: it differs from baseline only by using the hard-only
# prompt battery, so overriding it with the comprehensive battery would duplicate baseline.
policies=(baseline a1 a2c a3 a4 a5 a6-2 a6-4 a7-no-stop-guard replan-test)

declare -A resume_dirs=()

build_prompts "$prompts" easy_prompts.json medium_prompts.json hard_prompts.json

for policy in "${policies[@]}"; do
    resume_dir="${resume_dirs[$policy]:-}"
    if [[ -n "$resume_dir" ]]; then
        run_policy Resuming "$policy" --prompts "$prompts" --tries 3 \
            --output-dir "$resume_dir" --resume
    else
        run_policy Running "$policy" --prompts "$prompts" --tries 3 \
            --name "gem-context-ablation-$policy-comprehensive"
    fi
done

print_summary "Prompt manifest, logs, and status files"
