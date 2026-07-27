#!/usr/bin/env bash
set -euo pipefail

# Run this script from the Mac that can reach both 4090bld and hpc2.
# Flow: 4090bld:/.../franka -> Mac staging -> hpc2:/.../gensim/franka.

SRC_ALIAS="${SRC_ALIAS:-4090bld}"
HPC_ALIAS="${HPC_ALIAS:-hpc2}"
SRC_DIR="${SRC_DIR:-/home/bld/ykqin/InternDataEngine/output/simbox_plan_with_render/BananaBaseTask/franka}"
HPC_PARENT="${HPC_PARENT:-/hpc2hdd/home/yqin969/project/data/starvla/gensim}"
HPC_FINAL="${HPC_PARENT}/franka"
HPC_TMP="${HPC_PARENT}/.franka.transfer_tmp"
STAGE_ROOT="${STAGE_ROOT:-${HOME}/transfer_staging/simbox_BananaBaseTask_franka}"
STAGE_FINAL="${STAGE_ROOT}/franka"
EXCLUDE_DEMO_MP4="${EXCLUDE_DEMO_MP4:-0}"

SSH_OPTS=(
  -o ClearAllForwardings=yes
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=3
)
RSYNC_SSH="ssh -o ClearAllForwardings=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

DRY_RUN=0
PREFLIGHT_ONLY=0
CLEANUP_STAGE=1
ALLOW_NON_MAC=0

usage() {
  cat <<'EOF'
Usage:
  transfer_franka_to_hpc2_via_mac.sh [options]

Options:
  --dry-run         Run rsync in preview mode and do not move or clean anything.
  --preflight-only  Check SSH, rsync, disk space, and source counts, then exit.
  --keep-stage      Do not delete the Mac staging directory after success.
  --exclude-demo-mp4
                    Exclude preview videos named demo.mp4 from staging and HPC.
  --include-demo-mp4
                    Include preview videos named demo.mp4. This is the default.
  --allow-non-mac   Allow running outside macOS. Intended only for testing.
  -h, --help        Show this help.

Environment overrides:
  SRC_ALIAS=4090bld-direct   Override the source SSH alias.
  HPC_ALIAS=hpc2             Override the HPC SSH alias.
  SRC_DIR=/path/to/franka    Override the source franka path.
  HPC_PARENT=/path/to/gensim Override the HPC destination parent.
  STAGE_ROOT=/path/to/stage  Override the Mac staging root.
  EXCLUDE_DEMO_MP4=1         Exclude demo.mp4 preview videos.

Default behavior:
  1. Check Mac, source, and HPC prerequisites.
  2. Rsync non-fail franka data from 4090bld to Mac staging.
  3. Rsync Mac staging to hpc2 temporary directory.
  4. Verify counts and absence of fail_* directories.
  5. Atomically move the temporary directory into the final HPC path.
  6. Delete Mac staging only after the final HPC path verifies cleanly.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      ;;
    --keep-stage)
      CLEANUP_STAGE=0
      ;;
    --exclude-demo-mp4)
      EXCLUDE_DEMO_MP4=1
      ;;
    --include-demo-mp4)
      EXCLUDE_DEMO_MP4=0
      ;;
    --allow-non-mac)
      ALLOW_NON_MAC=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

q() {
  # Single-quote a string for remote POSIX shell use. Paths here should not
  # contain newlines.
  printf "%s" "$1" | sed "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/"
}

remote_src() {
  ssh -n "${SSH_OPTS[@]}" "$SRC_ALIAS" "$1"
}

remote_hpc() {
  ssh -n "${SSH_OPTS[@]}" "$HPC_ALIAS" "$1"
}

count_local_files() {
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    find "$1" -path '*/fail_*' -prune -o -type f ! -name 'demo.mp4' -print | wc -l | awk '{print $1}'
  else
    find "$1" -path '*/fail_*' -prune -o -type f -print | wc -l | awk '{print $1}'
  fi
}

local_fail_dir_sample() {
  find "$1" -type d -name 'fail_*' -print -quit
}

local_demo_sample() {
  find "$1" -path '*/fail_*' -prune -o -type f -name 'demo.mp4' -print -quit
}

remote_file_count_cmd() {
  local dir_q
  dir_q="$(q "$1")"
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    printf "find %s -path '*/fail_*' -prune -o -type f ! -name 'demo.mp4' -print | wc -l | awk '{print \$1}'" "$dir_q"
  else
    printf "find %s -path '*/fail_*' -prune -o -type f -print | wc -l | awk '{print \$1}'" "$dir_q"
  fi
}

remote_fail_sample_cmd() {
  local dir_q
  dir_q="$(q "$1")"
  printf "find %s -type d -name 'fail_*' -print -quit" "$dir_q"
}

remote_demo_sample_cmd() {
  local dir_q
  dir_q="$(q "$1")"
  printf "find %s -path '*/fail_*' -prune -o -type f -name 'demo.mp4' -print -quit" "$dir_q"
}

remote_source_bytes_cmd() {
  local dir_q
  dir_q="$(q "$SRC_DIR")"
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    printf "find %s -path '*/fail_*' -prune -o -type f ! -name 'demo.mp4' -printf '%%s\\n' | awk '{sum += \$1} END {print sum + 0}'" "$dir_q"
  else
    printf "find %s -path '*/fail_*' -prune -o -type f -printf '%%s\\n' | awk '{sum += \$1} END {print sum + 0}'" "$dir_q"
  fi
}

local_du_kb() {
  if [ -d "$1" ]; then
    du -sk "$1" | awk '{print $1}'
  else
    printf '0\n'
  fi
}

clean_local_fail_dirs() {
  if [ -d "$STAGE_FINAL" ]; then
    log "Removing stale fail_* directories from Mac staging, if any."
    find "$STAGE_FINAL" -type d -name 'fail_*' -prune -exec rm -rf {} +
  fi
}

clean_local_demo_files() {
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ] && [ -d "$STAGE_FINAL" ]; then
    log "Removing stale demo.mp4 preview videos from Mac staging, if any."
    find "$STAGE_FINAL" -path '*/fail_*' -prune -o -type f -name 'demo.mp4' -exec rm -f {} +
  fi
}

clean_hpc_tmp_fail_dirs() {
  local hpc_tmp_q
  hpc_tmp_q="$(q "$HPC_TMP")"
  remote_hpc "if test -d $hpc_tmp_q; then find $hpc_tmp_q -type d -name 'fail_*' -prune -exec rm -rf {} +; fi"
}

clean_hpc_tmp_demo_files() {
  local hpc_tmp_q
  hpc_tmp_q="$(q "$HPC_TMP")"
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    remote_hpc "if test -d $hpc_tmp_q; then find $hpc_tmp_q -path '*/fail_*' -prune -o -type f -name 'demo.mp4' -exec rm -f {} +; fi"
  fi
}

verify_local_stage() {
  local expected_count actual_count fail_sample demo_sample
  expected_count="$1"

  [ -d "$STAGE_FINAL" ] || die "Mac staging directory is missing: $STAGE_FINAL"

  actual_count="$(count_local_files "$STAGE_FINAL")"
  if [ "$actual_count" != "$expected_count" ]; then
    die "Mac staging file count mismatch: expected $expected_count, got $actual_count"
  fi

  fail_sample="$(local_fail_dir_sample "$STAGE_FINAL" || true)"
  if [ -n "$fail_sample" ]; then
    die "Mac staging still contains fail_* directory: $fail_sample"
  fi

  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    demo_sample="$(local_demo_sample "$STAGE_FINAL" || true)"
    if [ -n "$demo_sample" ]; then
      die "Mac staging still contains demo.mp4 although EXCLUDE_DEMO_MP4=1: $demo_sample"
    fi
  fi

  log "Mac staging verified: $actual_count files, no fail_* directories."
}

verify_remote_dir() {
  local label dir expected_count actual_count fail_sample demo_sample dir_q
  label="$1"
  dir="$2"
  expected_count="$3"
  dir_q="$(q "$dir")"

  remote_hpc "test -d $dir_q" || die "$label directory is missing on HPC: $dir"

  actual_count="$(remote_hpc "$(remote_file_count_cmd "$dir")")"
  if [ "$actual_count" != "$expected_count" ]; then
    die "$label file count mismatch on HPC: expected $expected_count, got $actual_count"
  fi

  fail_sample="$(remote_hpc "$(remote_fail_sample_cmd "$dir")" || true)"
  if [ -n "$fail_sample" ]; then
    die "$label still contains fail_* directory on HPC: $fail_sample"
  fi

  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    demo_sample="$(remote_hpc "$(remote_demo_sample_cmd "$dir")" || true)"
    if [ -n "$demo_sample" ]; then
      die "$label still contains demo.mp4 on HPC although EXCLUDE_DEMO_MP4=1: $demo_sample"
    fi
  fi

  log "$label verified on HPC: $actual_count files, no fail_* directories."
}

cleanup_stage_after_success() {
  if [ "$CLEANUP_STAGE" -ne 1 ]; then
    log "Keeping Mac staging because --keep-stage was provided: $STAGE_ROOT"
    return
  fi

  if [ -d "$STAGE_ROOT" ]; then
    log "Final HPC data verified. Removing Mac staging: $STAGE_ROOT"
    rm -rf "$STAGE_ROOT"
  fi
}

check_run_host() {
  local os_name
  os_name="$(uname -s)"
  if [ "$os_name" != "Darwin" ] && [ "$ALLOW_NON_MAC" -ne 1 ]; then
    die "This script is intended to run from your Mac. Current OS is $os_name. Use --allow-non-mac only if intentional."
  fi
}

preflight() {
  local src_dir_q hpc_parent_q source_count source_bytes source_kb
  local available_kb existing_stage_kb effective_available_kb required_kb

  src_dir_q="$(q "$SRC_DIR")"
  hpc_parent_q="$(q "$HPC_PARENT")"

  check_run_host

  command -v ssh >/dev/null 2>&1 || die "Local ssh is not available."
  command -v rsync >/dev/null 2>&1 || die "Local rsync is not available."
  command -v find >/dev/null 2>&1 || die "Local find is not available."
  case "$EXCLUDE_DEMO_MP4" in
    0|1)
      ;;
    *)
      die "EXCLUDE_DEMO_MP4 must be 0 or 1, got: $EXCLUDE_DEMO_MP4"
      ;;
  esac

  log "Checking source host: $SRC_ALIAS"
  remote_src "hostname; test -d $src_dir_q; command -v rsync"

  log "Checking HPC host and target parent: $HPC_ALIAS:$HPC_PARENT"
  remote_hpc "pwd; mkdir -p $hpc_parent_q; test -w $hpc_parent_q; df -h $hpc_parent_q; command -v rsync"

  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    log "Counting source files and bytes, excluding fail_* directories and demo.mp4 preview videos."
  else
    log "Counting source files and bytes, excluding fail_* directories. demo.mp4 preview videos are included."
  fi
  source_count="$(remote_src "$(remote_file_count_cmd "$SRC_DIR")")"
  source_bytes="$(remote_src "$(remote_source_bytes_cmd)")"
  [ "$source_count" -gt 0 ] || die "Source count is zero after excluding fail_* directories."

  source_kb=$(( (source_bytes + 1023) / 1024 ))
  available_kb="$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')"
  existing_stage_kb="$(local_du_kb "$STAGE_ROOT")"
  effective_available_kb=$((available_kb + existing_stage_kb))
  required_kb=$((source_kb + source_kb / 5 + 1024 * 1024))

  log "Source selected file count: $source_count"
  log "Source selected size estimate: $((source_kb / 1024)) MiB"
  log "Mac free space under HOME: $((available_kb / 1024)) MiB"
  log "Existing staging size counted as reusable: $((existing_stage_kb / 1024)) MiB"
  log "Required effective local space: $((required_kb / 1024)) MiB"

  if [ "$effective_available_kb" -lt "$required_kb" ]; then
    die "Mac does not appear to have enough effective space for staging. Need about $((required_kb / 1024)) MiB, have $((effective_available_kb / 1024)) MiB including existing staging."
  fi

  SOURCE_COUNT="$source_count"
}

main() {
  local source_count hpc_final_q hpc_tmp_q rsync_flags

  preflight
  source_count="$SOURCE_COUNT"
  hpc_final_q="$(q "$HPC_FINAL")"
  hpc_tmp_q="$(q "$HPC_TMP")"

  if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    log "Preflight finished. No transfer started."
    exit 0
  fi

  if remote_hpc "test -e $hpc_final_q"; then
    log "Final HPC path already exists: $HPC_FINAL"
    verify_remote_dir "Final path" "$HPC_FINAL" "$source_count"
    cleanup_stage_after_success
    log "Existing final path is complete. Nothing else to do."
    exit 0
  fi

  rsync_flags=(-avh --stats --progress --partial --append-verify --no-owner --no-group "--exclude=fail_*/")
  if [ "$EXCLUDE_DEMO_MP4" -eq 1 ]; then
    rsync_flags+=("--exclude=demo.mp4")
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    rsync_flags+=("--dry-run")
    log "Dry-run mode enabled. No final move or cleanup will be performed."
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "Dry-run will not create, move, clean, or delete staging data."
  else
    mkdir -p "$STAGE_ROOT"
    clean_local_fail_dirs
    clean_local_demo_files
  fi

  log "Syncing source data to Mac staging: $STAGE_FINAL"
  rsync "${rsync_flags[@]}" \
    -e "$RSYNC_SSH" \
    "${SRC_ALIAS}:${SRC_DIR}" \
    "$STAGE_ROOT/"

  if [ "$DRY_RUN" -eq 1 ]; then
    log "Dry-run finished after source-to-Mac rsync preview."
    exit 0
  fi

  verify_local_stage "$source_count"

  log "Preparing HPC temporary directory: $HPC_TMP"
  remote_hpc "mkdir -p $hpc_tmp_q"
  clean_hpc_tmp_fail_dirs
  clean_hpc_tmp_demo_files

  log "Syncing Mac staging to HPC temporary directory."
  rsync "${rsync_flags[@]}" \
    -e "$RSYNC_SSH" \
    "${STAGE_FINAL}/" \
    "${HPC_ALIAS}:${HPC_TMP}/"

  verify_remote_dir "Temporary path" "$HPC_TMP" "$source_count"

  log "Moving verified temporary directory into final HPC path."
  remote_hpc "test ! -e $hpc_final_q && mv $hpc_tmp_q $hpc_final_q"

  verify_remote_dir "Final path" "$HPC_FINAL" "$source_count"

  log "Core training-file samples on HPC:"
  remote_hpc "find $hpc_final_q -name meta_info.pkl | head; find $hpc_final_q -path '*/lmdb/data.mdb' | head"

  cleanup_stage_after_success
  log "Transfer complete: $HPC_ALIAS:$HPC_FINAL"
}

main "$@"
