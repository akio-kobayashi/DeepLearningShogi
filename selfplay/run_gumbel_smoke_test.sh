#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_gumbel_smoke_test.sh --model MODEL --roots ROOTS_HCP --out-dir OUT_DIR [options]

Required:
  --model PATH         TensorRT model path passed to selfplay
  --roots PATH         roots.hcp path
  --out-dir PATH       directory for hcpe3/log outputs

Optional:
  --selfplay PATH      selfplay binary path (default: selfplay/bin/selfplay)
  --gpu-id N           gpu id (default: 0)
  --batchsize N        batch size (default: 8)
  --nodes N            teacher nodes (default: 32)
  --playouts N         playouts per move (default: 32)
  --threads N          search threads (default: 1)
  --random N           random opening plies (default: 0)
  --min-move N         minimum move to save (default: 1)
  --skip-baseline      only run gumbel_root case
  --extra ARG          extra argument forwarded to selfplay; may be repeated

Example:
  ./selfplay/run_gumbel_smoke_test.sh \
    --model /path/to/model \
    --roots /path/to/roots.hcp \
    --out-dir /tmp/dlshogi-smoke
EOF
}

SELFPLAY_BIN="selfplay/bin/selfplay"
GPU_ID=0
BATCHSIZE=8
NODES=32
PLAYOUTS=32
THREADS=1
RANDOM_PLIES=0
MIN_MOVE=1
MODEL=""
ROOTS=""
OUT_DIR=""
SKIP_BASELINE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --roots)
      ROOTS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --selfplay)
      SELFPLAY_BIN="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --batchsize)
      BATCHSIZE="$2"
      shift 2
      ;;
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --playouts)
      PLAYOUTS="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --random)
      RANDOM_PLIES="$2"
      shift 2
      ;;
    --min-move)
      MIN_MOVE="$2"
      shift 2
      ;;
    --skip-baseline)
      SKIP_BASELINE=1
      shift
      ;;
    --extra)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$ROOTS" || -z "$OUT_DIR" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -x "$SELFPLAY_BIN" ]]; then
  echo "selfplay binary not found or not executable: $SELFPLAY_BIN" >&2
  exit 1
fi

if [[ ! -f "$ROOTS" ]]; then
  echo "roots.hcp not found: $ROOTS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.hcpe3"
  local log="$OUT_DIR/${name}.log"

  echo "== Running ${name} =="
  echo "log: ${log}"
  echo "output: ${output}"

  "$SELFPLAY_BIN" \
    --threads "$THREADS" \
    --random "$RANDOM_PLIES" \
    --min_move "$MIN_MOVE" \
    "$@" \
    "$MODEL" \
    "$ROOTS" \
    "$output" \
    "$NODES" \
    "$PLAYOUTS" \
    "$GPU_ID" \
    "$BATCHSIZE" \
    "${EXTRA_ARGS[@]}" \
    >"$log" 2>&1

  if [[ ! -s "$output" ]]; then
    echo "selfplay finished but output is empty: $output" >&2
    echo "tail of log:" >&2
    tail -n 40 "$log" >&2 || true
    exit 1
  fi

  wc -c "$output"
  tail -n 20 "$log" || true
}

if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  run_case baseline
fi

run_case gumbel_root \
  --gumbel_root \
  --gumbel_scale 1.0 \
  --gumbel_prior_weight 1.0 \
  --gumbel_value_weight 1.0 \
  --gumbel_visit_weight 0.25 \
  --gumbel_export_temperature 1.0

echo "Smoke test completed. Outputs are under: $OUT_DIR"
