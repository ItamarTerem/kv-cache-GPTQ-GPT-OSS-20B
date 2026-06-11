#!/usr/bin/env bash
# scripts/download_model.sh
#
# Download GPT-OSS 20B weights from HuggingFace.
#
# Supported models (--model_name):
#   openai   → openai/gpt-oss-20b          (https://huggingface.co/openai/gpt-oss-20b)
#   bf16     → unsloth/gpt-oss-20b-BF16    (https://huggingface.co/unsloth/gpt-oss-20b-BF16)
#
# Usage:
#   bash scripts/download_model.sh --model_name openai
#   bash scripts/download_model.sh --model_name bf16
#
# Overrides:
#   MODEL_DIR  — local save path       (default: <repo_root>/models/<model_name>)
#   REVISION   — git revision / branch (default: main)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse --model_name argument ───────────────────────────────────────────────
MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/download_model.sh --model_name <openai|bf16>"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: --model_name is required."
    echo "Usage: bash scripts/download_model.sh --model_name <openai|bf16>"
    exit 1
fi

case "$MODEL_NAME" in
    openai)
        DEFAULT_MODEL_ID="openai/gpt-oss-20b"
        DEFAULT_MODEL_DIR="${ROOT}/models/gpt-oss-20b"
        ;;
    bf16)
        DEFAULT_MODEL_ID="unsloth/gpt-oss-20b-BF16"
        DEFAULT_MODEL_DIR="${ROOT}/models/gpt-oss-20b-BF16"
        ;;
    *)
        echo "ERROR: Unknown model_name '$MODEL_NAME'."
        echo "Valid options: openai, bf16"
        exit 1
        ;;
esac

MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
REVISION="${REVISION:-main}"

# Safety margin over the ~40 GB actual size
APPROX_SIZE_GIB=45

echo "======================================"
echo " GPT-OSS 20B downloader"
echo " MODEL_NAME: $MODEL_NAME"
echo " MODEL_ID  : $MODEL_ID"
echo " MODEL_DIR : $MODEL_DIR"
echo " REVISION  : $REVISION"
echo " Est. size : ~${APPROX_SIZE_GIB} GiB (~40 GB)"
echo "======================================"

# ── Check hf CLI ─────────────────────────────────────────────────────────────
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found."
    echo "Activate your venv and ensure huggingface_hub is installed:"
    echo "  source .venv/bin/activate"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# ── Disk space check ─────────────────────────────────────────────────────────
PARENT_DIR="$(dirname "$MODEL_DIR")"
mkdir -p "$PARENT_DIR"

AVAILABLE_GIB=$(df -BG "$PARENT_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Available disk space at $PARENT_DIR: ${AVAILABLE_GIB} GiB"

if (( AVAILABLE_GIB < APPROX_SIZE_GIB )); then
    echo ""
    echo "ERROR: Insufficient disk space."
    echo "  Required : ~${APPROX_SIZE_GIB} GiB"
    echo "  Available: ${AVAILABLE_GIB} GiB"
    echo ""
    echo "Free up space or point MODEL_DIR at a path with enough capacity:"
    echo "  MODEL_DIR=/path/to/large/disk bash scripts/download_model.sh"
    exit 1
fi
echo "✔ Disk space OK"

# ── HF token note ─────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "NOTE: HF_TOKEN is not set."
    if [[ "$MODEL_NAME" == "openai" ]]; then
        echo "openai/gpt-oss-20b may be a gated repo — if you see 401 errors, run:"
    else
        echo "unsloth/gpt-oss-20b-BF16 is public — no token needed."
        echo "If you see 401 errors, run:"
    fi
    echo "  hf auth login"
fi

# ── Enable fast transfer if hf_transfer is installed ─────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Prepare destination ───────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Starting download — ~40 GB, ETA depends on connection speed..."
echo ""

CMD=(
    hf download "$MODEL_ID"
    --local-dir "$MODEL_DIR"
    --revision "$REVISION"
)

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

# ── Verify download ───────────────────────────────────────────────────────────
echo ""
echo "Verifying downloaded files..."
WEIGHT_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
CONFIG_OK=$([ -f "$MODEL_DIR/config.json" ] && echo "yes" || echo "no")
TOKENIZER_OK=$([ -f "$MODEL_DIR/tokenizer.json" ] || [ -f "$MODEL_DIR/tokenizer_config.json" ] && echo "yes" || echo "no")

echo "  .safetensors shards : $WEIGHT_COUNT"
echo "  config.json         : $CONFIG_OK"
echo "  tokenizer files     : $TOKENIZER_OK"

if [[ "$CONFIG_OK" != "yes" ]]; then
    echo "WARNING: config.json missing — download may be incomplete."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Download complete ✔"
echo " Weights saved to: $MODEL_DIR"
echo ""
echo " Run the quantization pipeline:"
echo "   source .venv/bin/activate"
echo "   python test/stage1_nvfp4_unit_tests.py"
echo "   python test/stage2_nvfp4_algorithm_tests.py"
echo "   python test/stage3_gpt_oss_shape_tests.py"
echo "   python test/stage4_baseline_perplexity.py      \\"
echo "     --model_path $MODEL_DIR"
echo "   python test/stage5_quantize_model.py           \\"
echo "     --model_path $MODEL_DIR"
echo "   python test/stage6_eval_perplexity.py          \\"
echo "     --model_path $MODEL_DIR"
echo "======================================"