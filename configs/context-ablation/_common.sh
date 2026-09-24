# Shared helpers for the context-ablation runners. Source after setting script_dir.

repo_root=$(cd -- "$script_dir/../.." && pwd)
run_stamp=$(date +%Y%m%d_%H%M%S)
statuses=()
failed=0

trap 'exit 130' INT TERM

# build_prompts DEST BATTERY...: merge sari_bench/prompts batteries into DEST (kept next to the logs,
# so prompts added to any battery are picked up at launch). Exits on failure.
build_prompts() {
    local destination=$1
    shift
    local sources=()
    local battery
    for battery in "$@"; do
        sources+=("$repo_root/sari_bench/prompts/$battery")
    done
    (
        cd "$repo_root"
        uv run python - "$destination" "${sources[@]}" <<'PY'
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
    local merge_status=$?
    if (( merge_status != 0 )); then
        printf 'Failed to build the comprehensive prompt battery (status %s).\n' "$merge_status" >&2
        exit "$merge_status"
    fi
}

# run_policy LABEL POLICY [ARGS...]: run POLICY's config with extra sari_bench args, logging to
# $log_dir and recording its exit status.
run_policy() {
    local label=$1 policy=$2
    shift 2
    local log="$log_dir/$policy.log"
    printf '%s %-*s -> %s\n' "$label" $(( 25 - ${#label} )) "$policy" "$log"
    (
        cd "$repo_root"
        uv run python -m sari_bench run --config "$script_dir/$policy.toml" "$@"
    ) >"$log" 2>&1
    local status=$?
    printf '%s\n' "$status" >"$log_dir/$policy.status"
    statuses+=("$status")
    if (( status != 0 )); then
        failed=1
    fi
}

# print_summary FOOTER: per-policy status table, then exit with the aggregate status.
print_summary() {
    printf '\n%-18s %s\n' "CONFIGURATION" "STATUS"
    printf '%-18s %s\n' "------------------" "------"
    local index status label
    for index in "${!policies[@]}"; do
        status=${statuses[$index]}
        label="ok"
        if (( status != 0 )); then
            label="failed ($status)"
        fi
        printf '%-18s %s\n' "${policies[$index]}" "$label"
    done
    printf '\n%s: %s\n' "$1" "$log_dir"
    exit "$failed"
}
