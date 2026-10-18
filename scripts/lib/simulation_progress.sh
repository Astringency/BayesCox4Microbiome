#!/usr/bin/env bash
# Shared logging/queue display only; experiment arguments stay in each launcher.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

simulation_setup() {
    SIM_EXTRA=(); SIM_NAMES=(); SIM_LOGS=(); SIM_STATUS=(); SIM_COUNTS=(); SIM_TOTALS=(); SIM_PIDS=()
    SIM_STAMP="$(date +%Y%m%d_%H%M%S)_$$"; SIM_LINES=0; SIM_LAST=''
    while (( $# )); do
        case "${1%%=*}" in
            --help) echo 'Set grid/budget in this script or N_VALUES,P_VALUES,NONZERO_VALUES,BASELINES,REPS,PRIORS,WORKERS,OUT_ROOT. DRY_RUN=true previews; detailed output goes to OUT_ROOT/logs/. Extra Julia options are forwarded.'; exit 0;;
            --n|--p|--nonzero|--baseline|--constraint|--priors|--outdir|--reps)
                local key="${1%%=*}" value
                if [[ "$1" == *=* ]]; then value="${1#*=}"; else
                    (( $# >= 2 )) && [[ "$2" != --* ]] || { echo "$1 needs a value" >&2; exit 2; }
                    value="$2"; shift
                fi
                case "$key" in
                    --n) N_VALUES="$value";; --p) P_VALUES="$value";; --nonzero) NONZERO_VALUES="$value";;
                    --baseline) BASELINES="$value";; --constraint) CONSTRAINTS="$value";; --priors) PRIORS="$value";;
                    --outdir) OUT_ROOT="$value";; --reps) REPS="$value";;
                esac;;
            --progress-file|--progress-file=*) echo '--progress-file is managed by the launcher' >&2; exit 2;;
            *) SIM_EXTRA+=("$1");;
        esac
        shift
    done
    [[ "${WORKERS:-1}" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be positive' >&2; exit 2; }
    if [[ "${DRY_RUN:-false}" != true ]]; then
        SIM_TMP=$(mktemp -d); trap simulation_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
    fi
}

simulation_cleanup() {
    local pid
    for pid in "${SIM_PIDS[@]}"; do
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    for pid in "${SIM_PIDS[@]}"; do [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; done
    [[ -z "${SIM_TMP:-}" ]] || rm -rf -- "$SIM_TMP"
}

simulation_job() {
    local label="$1" outdir="$2"; shift 2
    local id=${#SIM_NAMES[@]} reps=1 priors='' rep_ids='' i
    local args=("$@" "${SIM_EXTRA[@]}")
    for ((i=0;i<${#args[@]};i++)); do
        case "${args[i]}" in
            --reps) reps="${args[i+1]}";; --reps=*) reps="${args[i]#*=}";;
            --priors) priors="${args[i+1]}";; --priors=*) priors="${args[i]#*=}";;
            --rep-ids) rep_ids="${args[i+1]}";; --rep-ids=*) rep_ids="${args[i]#*=}";;
        esac
    done
    local methods=() ids=(); IFS=',' read -ra methods <<< "$priors"
    if [[ -n "$rep_ids" ]]; then IFS=',' read -ra ids <<< "$rep_ids"; reps=${#ids[@]}; fi
    [[ "$reps" =~ ^[1-9][0-9]*$ && ${#methods[@]} -gt 0 ]] || { echo 'Invalid repetitions/priors' >&2; return 2; }
    local log="${LOG_ROOT:-$OUT_ROOT/logs}/${label}_${SIM_STAMP}.log"
    SIM_NAMES+=("$label [$priors]"); SIM_LOGS+=("$log"); SIM_STATUS+=("等待")
    SIM_COUNTS+=(0); SIM_TOTALS+=("$((reps * ${#methods[@]}))"); SIM_PIDS+=('')
    if [[ "${DRY_RUN:-false}" == true ]]; then
        run_julia scripts/lib/simulation/run_simulation.jl --outdir "$outdir" "${args[@]}"
        printf '  log: %s\n' "$log"; return
    fi
    mkdir -p "$(dirname "$log")"
    printf 'set -e\n' > "$SIM_TMP/$id.sh"
    if [[ -n "${SIM_JOB_LOCK:-}" ]]; then
        mkdir -p "$(dirname "$SIM_JOB_LOCK")"
        printf 'exec 9> %q\nflock 9\n' "$SIM_JOB_LOCK" >> "$SIM_TMP/$id.sh"
    fi
    printf 'exec ' >> "$SIM_TMP/$id.sh"
    printf '%q ' "${BCX_JULIA[@]}" -t "${THREADS:-${CHAINS:-4}}" --startup-file=no \
        "--project=$BCX_PROJECT_ROOT" scripts/lib/simulation/run_simulation.jl --outdir "$outdir" \
        "${args[@]}" --progress-file "$log.progress.tsv" >> "$SIM_TMP/$id.sh"
    printf '\n' >> "$SIM_TMP/$id.sh"
}

simulation_display() {
    local total=0 count=0 done=0 failed=0 i snapshot='' row width=0 rows=24 cols=80
    for ((i=0;i<${#SIM_NAMES[@]};i++)); do
        total=$((total+SIM_TOTALS[i])); count=$((count+SIM_COUNTS[i]))
        [[ "${SIM_STATUS[i]}" != 完成 ]] || done=$((done+1))
        [[ "${SIM_STATUS[i]}" != 失败* ]] || failed=$((failed+1))
        printf -v row '  [%-6s] %3d/%-3d (%3d%%) %s\n' "${SIM_STATUS[i]}" "${SIM_COUNTS[i]}" \
            "${SIM_TOTALS[i]}" "$((100*SIM_COUNTS[i]/SIM_TOTALS[i]))" "${SIM_NAMES[i]}"
        snapshot+="$row"
        (( ${#row}+2 <= width )) || width=$((${#row}+2))
    done
    local title="总进度: $count/$total 拟合 ($((total ? 100*count/total : 0))%) | 完成 $done/${#SIM_NAMES[@]} 组 | 失败 $failed 组"
    [[ "$title$snapshot" != "$SIM_LAST" ]] || return 0
    # Redraw only when the entire table fits; otherwise retain readable snapshots.
    local redraw=false
    if [[ -t 1 ]]; then
        read -r rows cols < <(stty size < /dev/tty 2>/dev/null || echo '24 80')
        if (( ${#SIM_NAMES[@]}+1 < rows && width < cols && ${#title}+11 < cols )); then redraw=true; fi
    fi
    if [[ "$redraw" == true ]]; then
        (( SIM_LINES == 0 )) || printf '\033[%dA\033[J' "$SIM_LINES"
    fi
    printf '%s\n%s' "$title" "$snapshot"
    SIM_LINES=0; [[ "$redraw" != true ]] || SIM_LINES=$((${#SIM_NAMES[@]}+1))
    SIM_LAST="$title$snapshot"
}

simulation_finish() {
    [[ "${DRY_RUN:-false}" != true ]] || return 0
    local next=0 active=0 finished=0 failed=0 i code count total
    printf '详细采样输出: %s\n' "${LOG_ROOT:-$OUT_ROOT/logs}"
    while (( finished < ${#SIM_NAMES[@]} )); do
        while (( next < ${#SIM_NAMES[@]} && active < ${WORKERS:-1} )); do
            bash "$SIM_TMP/$next.sh" > "${SIM_LOGS[next]}" 2>&1 &
            SIM_PIDS[next]=$!; SIM_STATUS[next]='运行'; active=$((active+1)); next=$((next+1))
        done
        for ((i=0;i<next;i++)); do
            [[ -n "${SIM_PIDS[i]}" ]] || continue
            if [[ -s "${SIM_LOGS[i]}.progress.tsv" ]]; then
                read -r count total < "${SIM_LOGS[i]}.progress.tsv" || continue
                if [[ "$count" =~ ^[0-9]+$ && "$total" =~ ^[1-9][0-9]*$ ]]; then
                    SIM_COUNTS[i]=$count; SIM_TOTALS[i]=$total
                fi
            fi
            if ! kill -0 "${SIM_PIDS[i]}" 2>/dev/null; then
                if wait "${SIM_PIDS[i]}"; then
                    SIM_STATUS[i]='完成'; SIM_COUNTS[i]=${SIM_TOTALS[i]}
                else code=$?; SIM_STATUS[i]="失败($code)"; failed=1; fi
                SIM_PIDS[i]=''; active=$((active-1)); finished=$((finished+1))
            fi
        done
        simulation_display
        if (( finished < ${#SIM_NAMES[@]} )); then sleep "${PROGRESS_INTERVAL:-2}"; fi
    done
    return "$failed"
}
