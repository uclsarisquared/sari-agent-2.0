#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_stamp=$(date +%Y%m%d_%H%M%S)
log_dir="$script_dir/logs/$run_stamp-comprehensive"
prompts="$log_dir/comprehensive_prompts.json"
mkdir -p "$log_dir"

# baseline and a1 already finished in a prior comprehensive run; this picks up where that left off.
# hard-baseline is intentionally omitted: it differs from baseline only by using the hard-only
# prompt battery, so overriding it with the comprehensive battery would duplicate baseline.
policies=(a2c a3 a4 a5 a6-2 a6-4 a7-no-stop-guard)

prompt_files=(
    "$repo_root/sari_bench/prompts/easy_prompts.json"
    "$repo_root/sari_bench/prompts/medium_prompts.json"
    "$repo_root/sari_bench/prompts/hard_prompts.json"
)
statuses=()
failed=0

trap 'exit 130' INT TERM

# Keep the exact merged battery next to the logs. Building it at launch means newly added prompts
# in any of the three canonical difficulty batteries are included automatically.
(
    cd "$repo_root"
    uv run python - "$prompts" "${prompt_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1])
entries = []
seen = set()

for source_arg in sys.argv[2:]:
    source = Path(source_arg)
    raw = json.loads(source.read_text(encoding="utf-8"))
    prompts = raw.get("prompts") if isinstance(raw, dict) else raw
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"{source} contains no prompts")
    for prompt in prompts:
        prompt_id = prompt.get("id") if isinstance(prompt, dict) else None
        if not prompt_id:
            raise SystemExit(f"{source} contains a prompt without an id")
        if prompt_id in seen:
            raise SystemExit(f"duplicate prompt id across batteries: {prompt_id}")
        seen.add(prompt_id)
        entries.append(prompt)

destination.write_text(
    json.dumps({"prompts": entries}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Built {destination} with {len(entries)} prompts")
PY
)
merge_status=$?
if (( merge_status != 0 )); then
    printf 'Failed to build the comprehensive prompt battery (status %s).\n' "$merge_status" >&2
    exit "$merge_status"
fi

for policy in "${policies[@]}"; do
    config="$script_dir/$policy.toml"
    log="$log_dir/$policy.log"
    status_file="$log_dir/$policy.status"
    name="context-ablation-$policy-comprehensive"
    printf 'Running %-18s -> %s\n' "$policy" "$log"
    (
        cd "$repo_root"
        uv run python -m sari_bench run \
            --config "$config" \
            --prompts "$prompts" \
            --tries 3 \
            --name "$name"
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
printf '\nPrompt manifest, logs, and status files: %s\n' "$log_dir"

exit "$failed"
