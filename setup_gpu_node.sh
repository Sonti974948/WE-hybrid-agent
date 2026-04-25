#!/bin/bash
# ============================================================
# setup_gpu_node.sh
# Sets up WE-Hybrid Terminal Agent on EXPANSE GPU node
# No root/sudo required. No Ollama — uses HuggingFace directly.
#
# RUN THIS ON THE EXPANSE LOGIN NODE (has internet):
#   bash setup_gpu_node.sh
#   bash setup_gpu_node.sh --model Qwen/Qwen2.5-7B-Instruct   # no HF token needed
#   bash setup_gpu_node.sh --no-download                       # skip model download
#
# Then get a GPU node and run the agent:
#   sinteractive --partition=gpu-shared --gpus=1 --time=4:00:00 --account=YOUR_ACCOUNT
#   cd ~/We_hybrid/we_wizard/terminal_agent
#   python run_agent.py
# ============================================================

set -e

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SKIP_DOWNLOAD=false
QUANTIZATION="8bit"

# Parse args
for arg in "$@"; do
    case $arg in
        --model=*)      MODEL="${arg#*=}" ;;
        --model)        shift; MODEL="$1" ;;
        --quantization=*) QUANTIZATION="${arg#*=}" ;;
        --no-download)  SKIP_DOWNLOAD=true ;;
    esac
done

echo "================================================"
echo "  WE-Hybrid Agent — EXPANSE Setup"
echo "  Model:         $MODEL"
echo "  Quantization:  $QUANTIZATION"
echo "================================================"
echo ""

# ── 1. Check HF token for gated models ───────────────────────────────────────
LLAMA_MODELS=("meta-llama" "mistralai" "google/gemma")
IS_GATED=false
for prefix in "${LLAMA_MODELS[@]}"; do
    if [[ "$MODEL" == "$prefix"* ]]; then
        IS_GATED=true
        break
    fi
done

if $IS_GATED; then
    if [ -z "$HF_TOKEN" ]; then
        echo "⚠  Model '$MODEL' is gated and requires a HuggingFace token."
        echo ""
        echo "  Steps:"
        echo "    1. Create account: https://huggingface.co"
        echo "    2. Accept license: https://huggingface.co/$MODEL"
        echo "    3. Get token:      https://huggingface.co/settings/tokens"
        echo "    4. Run: export HF_TOKEN=hf_xxxxxxxxxxxx"
        echo "    5. Re-run this script"
        echo ""
        echo "  Or use a model that needs no token:"
        echo "    bash setup_gpu_node.sh --model Qwen/Qwen2.5-7B-Instruct"
        echo ""
        read -p "  Enter HF_TOKEN now (or press Enter to skip model download): " HF_TOKEN
    fi

    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
        echo "✓ HF_TOKEN set"
    fi
fi

# ── 2. Install Python packages ────────────────────────────────────────────────
echo ""
echo "Installing Python packages..."

CORE_PKGS="langgraph langchain-core langchain-huggingface"
TORCH_PKGS="transformers accelerate bitsandbytes"
UI_PKGS="rich pydantic"
RAG_PKGS="sentence-transformers chromadb"

pip install $CORE_PKGS $TORCH_PKGS $UI_PKGS $RAG_PKGS \
    --quiet --break-system-packages 2>/dev/null || \
pip install $CORE_PKGS $TORCH_PKGS $UI_PKGS $RAG_PKGS \
    --quiet --user 2>/dev/null || \
echo "  pip install failed — try: conda install or pip install ... --user"

# ── 3. Sanity check packages ──────────────────────────────────────────────────
echo ""
echo "Checking packages..."
python3 -c "
import importlib, sys
core = ['langgraph', 'langchain_core', 'langchain_huggingface',
        'transformers', 'accelerate', 'rich', 'pydantic']
rag  = ['sentence_transformers', 'chromadb']
ok, warn = [], []
for pkg in core:
    try: importlib.import_module(pkg); ok.append(pkg)
    except ImportError: print(f'  ERROR: missing {pkg}')
for pkg in rag:
    try: importlib.import_module(pkg); ok.append(pkg)
    except ImportError: warn.append(pkg)
if warn:
    print(f'  WARNING: RAG packages missing: {warn}')
    print('  Agent works without RAG, but install for Q&A: pip install sentence-transformers chromadb --user')
try:
    import torch
    print(f'  ✓ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('  ERROR: torch not found — CUDA acceleration unavailable')
print(f'  ✓ {len(ok)} packages OK')
"

# ── 4. Download model weights ─────────────────────────────────────────────────
if ! $SKIP_DOWNLOAD; then
    echo ""
    echo "Downloading model: $MODEL"
    echo "(Downloads to ~/.cache/huggingface/ — reused on every GPU node)"
    echo ""

    HF_ARGS=""
    [ -n "$HF_TOKEN" ] && HF_ARGS="--token $HF_TOKEN"

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -c "
from huggingface_hub import snapshot_download
import os, sys
model_id = '$MODEL'
token = os.environ.get('HF_TOKEN') or None
print(f'Downloading {model_id}...')
try:
    path = snapshot_download(model_id, token=token, ignore_patterns=['*.gguf', '*.ot'])
    print(f'✓ Model cached at: {path}')
except Exception as e:
    print(f'✗ Download failed: {e}')
    if '401' in str(e) or 'gated' in str(e).lower():
        print()
        print('  This model is gated. You need to:')
        print(f'  1. Accept license at: https://huggingface.co/{model_id}')
        print('  2. Set HF_TOKEN and re-run')
    sys.exit(1)
"
    else
        echo "  huggingface_hub not found, trying huggingface-cli..."
        huggingface-cli download "$MODEL" $HF_ARGS || \
            echo "  Download failed. The agent will download automatically on first run."
    fi
else
    echo "Skipping model download (--no-download). Will download on first run."
fi

# ── 5. Build RAG index (optional) ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if python3 -c "import sentence_transformers, chromadb" 2>/dev/null; then
    if [ ! -d "$HOME/.westpa_index" ]; then
        echo ""
        echo "Building RAG documentation index (one-time, ~2-5 min)..."
        # Auto-detect repo dir
        REPO_DIR=""
        for candidate in "$HOME/We_hybrid" "$HOME/ParMetaD" "$(dirname "$SCRIPT_DIR")/../.."; do
            if [ -d "$candidate" ]; then
                REPO_DIR="$candidate"
                break
            fi
        done

        if [ -n "$REPO_DIR" ]; then
            python3 "$SCRIPT_DIR/build_index.py" --repo-dir "$REPO_DIR" --no-test || \
                echo "  RAG index build failed — agent will still work without it"
        else
            python3 "$SCRIPT_DIR/build_index.py" --no-test || \
                echo "  RAG index build failed — agent will still work without it"
        fi
    else
        echo "✓ RAG index already exists at ~/.westpa_index"
    fi
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  To run the agent (on a GPU node):"
echo "    sinteractive --partition=gpu-shared --gpus=1 --time=4:00:00 --account=YOUR_ACCOUNT"
echo "    cd ~/We_hybrid/we_wizard/terminal_agent"
echo "    python run_agent.py"
echo ""
echo "  Model options:"
echo "    --model meta-llama/Llama-3.1-8B-Instruct  [default, best quality]"
echo "    --model Qwen/Qwen2.5-7B-Instruct          [no HF token needed]"
echo "    --model meta-llama/Llama-3.2-3B-Instruct  [faster, smaller]"
echo ""
echo "  Quantization options (--quantization):"
echo "    8bit   ~8GB VRAM  [default]"
echo "    4bit   ~4GB VRAM  [less accurate]"
echo "    fp16   ~16GB VRAM [best quality]"
echo "================================================"
