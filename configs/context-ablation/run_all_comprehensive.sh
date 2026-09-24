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

# Resume targets as "policy=battery_dir" entries (a plain array: macOS bash 3.2 lacks declare -A).
resume_dirs=()

resume_dir_for() {
    local entry
    for entry in ${resume_dirs[@]+"${resume_dirs[@]}"}; do
        if [[ "${entry%%=*}" == "$1" ]]; then
            printf '%s' "${entry#*=}"
            return
        fi
    done
}

build_prompts "$prompts" easy_prompts.json medium_prompts.json hard_prompts.json

for policy in "${policies[@]}"; do
    resume_dir=$(resume_dir_for "$policy")
    if [[ -n "$resume_dir" ]]; then
        # --name "" clears the config's name, which sari_bench rejects alongside --output-dir.
        run_policy Resuming "$policy" --prompts "$prompts" --tries 3 \
            --name "" --output-dir "$resume_dir" --resume
    else
        run_policy Running "$policy" --prompts "$prompts" --tries 3 \
            --name "gem-context-ablation-$policy-comprehensive"
    fi
done

print_summary "Prompt manifest, logs, and status files"
