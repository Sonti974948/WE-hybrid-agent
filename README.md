# WE-Hybrid Agent — HuggingFace Terminal Agent

A local LLM terminal agent for setting up **ParMetaD** and **ParGaMD** weighted ensemble simulations. Runs entirely on your HPC cluster GPU — no external API or internet connection required after setup.

Built with **LangGraph** + **HuggingFace Transformers** + **ChromaDB RAG**.

---

## Overview

```
run_agent.py          # Rich terminal UI — start here
llm_agent.py          # LangGraph agent, HuggingFace LLM backend
rag.py                # ChromaDB + sentence-transformers RAG pipeline
build_index.py        # One-time RAG index builder (run on login node)
config_schema.py      # Pydantic models for simulation parameters
generators.py         # Generates all WE simulation input files
setup_gpu_node.sh     # Installs all dependencies on EXPANSE GPU node
requirements.txt      # Python dependencies
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| HPC cluster with NVIDIA GPU | V100 or better; A100 recommended |
| Python 3.10+ | Via conda (module load anaconda3) |
| ~10 GB free disk | For model weights + RAG index |
| HuggingFace account | Free. Token needed only for gated models (Llama) |
| AMBER (optional) | For structure prep (tleap). Templates generated without it |

> **Note:** The agent was tested on EXPANSE (SDSC) with V100 GPUs. A 3B–7B model at 8-bit quantization requires ~6–8 GB VRAM.

---

## Installation on EXPANSE

### Step 1 — Get an interactive GPU node

```bash
sinteractive --partition=gpu-shared --gpus=1 --time=4:00:00 --account=YOUR_ACCOUNT
```

### Step 2 — Clone and install

```bash
git clone https://github.com/Sonti974948/WE-hybrid-agent.git
cd WE-hybrid-agent
git checkout hf-agent

bash setup_gpu_node.sh
```

`setup_gpu_node.sh` will:
- Load required modules (`gpu`, `anaconda3`)
- Install all Python dependencies into your active conda environment
- Download model weights via HuggingFace hub (you will be prompted for model choice)
- Build the RAG index from WESTPA documentation and local files

### Step 3 — (Optional) Build RAG index manually

```bash
# From the login node (internet access):
python build_index.py --index-dir ./rag_index

# To also index your own simulation files:
python build_index.py --index-dir ./rag_index --repo-dir /path/to/your/We_hybrid
```

---

## Running the Agent

```bash
# Basic — uses default model (Qwen2.5-3B, 8-bit)
python run_agent.py

# Specify model and quantization
python run_agent.py --model Qwen/Qwen2.5-7B-Instruct --quantization 8bit

# With HuggingFace token (required for gated models like Llama)
python run_agent.py --model meta-llama/Llama-3.2-3B-Instruct --hf-token hf_xxx

# Set output directory for generated files
python run_agent.py --output ./my_simulation

# Fallback: dummy LLM for testing without GPU
python run_agent.py --fallback
```

### Recommended models by GPU

| GPU VRAM | Recommended model | Quantization |
|----------|-------------------|-------------|
| 8 GB (V100 16 GB safe) | `Qwen/Qwen2.5-3B-Instruct` | 8bit |
| 16 GB | `Qwen/Qwen2.5-7B-Instruct` | 8bit |
| 40 GB (A100) | `meta-llama/Llama-3.1-8B-Instruct` | fp16 |
| 80 GB (A100) | `meta-llama/Llama-3.1-70B-Instruct` | 4bit |

---

## How the Agent Works

1. **Startup** — checks GPU, loads HF model, verifies RAG index
2. **Conversation** — you describe your system; agent collects parameters
3. **`<params>` tracking** — agent embeds JSON in every response:
   ```
   <params>{"method": "parmetad", "backend": "amber", "nsteps": 50000}</params>
   ```
   Python parses and accumulates these silently — no tool-calling required
4. **Validation** — checks pcoord_len constraint (`nsteps / ntpr + 1`) and other rules
5. **File generation** — once all parameters collected, writes all simulation files to `--output` directory

---

## Parameters the Agent Collects

| Parameter | Example | Notes |
|-----------|---------|-------|
| `method` | `parmetad` | parmetad or parGaMD |
| `backend` | `amber` | amber or openmm |
| `scratch_dir` | `/expanse/lustre/scratch/user/proj` | Lustre scratch, NOT home dir |
| `conda_env` | `westpa2` | Must have WESTPA installed |
| `account` | `ucd192` | SLURM billing project |
| `partition` | `gpu-shared` | EXPANSE: gpu-shared or gpu |
| `nodes` | `4` | Each node runs gpus_per_node walkers |
| `gpus_per_node` | `4` | Total walkers = nodes × gpus_per_node |
| `walltime` | `24:00:00` | WE checkpoints each iteration |
| `amberhome` | `/home/user/amber22` | AMBER installation path |
| `temperature` | `300` | Kelvin |
| `nsteps` | `50000` | MD steps per WE segment |
| `ntpr` | `500` | Output every ntpr steps |
| `bin_boundaries` | `[0,1,2,3,4,5]` | Progress coordinate bin edges |
| `bin_target_counts` | `4` | Walkers maintained per bin |
| `max_iterations` | `200` | Total WE iterations to run |

### Critical constraint

```
pcoord_len = nsteps / ntpr + 1
```

`nsteps` must be **exactly divisible** by `ntpr`. The agent checks this automatically.

---

## Generated Files

After all parameters are collected, the agent writes to `--output`:

```
west.cfg               # WESTPA configuration
runseg.sh              # Segment propagation script
run_WE.sh              # Main SLURM submission script
node.sh                # Worker node startup
env.sh                 # Environment setup
md.in                  # AMBER MD input
plumed.dat             # PLUMED metadynamics (ParMetaD only)
cMD/run_cmd.sh         # cGaMD pre-run script (ParGaMD only)
SETUP_INSTRUCTIONS.md  # Step-by-step launch guide
config_used.json       # Full parameter record
```

---

## RAG Knowledge Base

The agent uses a local ChromaDB vector store built from:
- WESTPA GitHub documentation
- PLUMED manual excerpts
- 11 hand-written expert docs covering: binning strategy, GaMD parameters, PLUMED metadynamics, EXPANSE setup, pcoord_len, debugging common errors

The RAG context is injected automatically before each LLM call — no internet needed at inference time.

---

## Troubleshooting

**Agent is too slow:**
- Use a smaller model: `Qwen/Qwen2.5-1.5B-Instruct` at 8bit
- Ask shorter, specific questions rather than broad explanations
- The V100 is slow for 7B+ models; request an A100 node if available:
  ```bash
  sinteractive --partition=gpu-shared --gpus=1 --time=4:00:00 \
    --constraint="gpu80" --account=YOUR_ACCOUNT
  ```

**Parameters not updating:**
- The LLM must include a `<params>{...}</params>` block. If it doesn't, re-prompt with:
  *"Set method to parmetad and backend to amber"*
- The agent tracks which fields are still missing and shows a countdown

**Model fails to load:**
- Gated models (Llama) require `--hf-token hf_xxx`
- First run downloads weights — ensure ~10 GB free in your home/scratch
- Try `--quantization 4bit` if VRAM is tight

**RAG index empty:**
- Run `python build_index.py` from the login node (internet access needed)
- Index is stored in `./rag_index/` — copy it to your GPU node scratch

---

## EXPANSE-Specific Notes

```bash
# Always run simulations from Lustre scratch — home dir NFS is too slow
scratch_dir=/expanse/lustre/scratch/$USER/your_project
mkdir -p $scratch_dir

# Check VRAM on your GPU node
nvidia-smi

# Check job status
squeue -u $USER

# Interactive A100 node (if available)
sinteractive --partition=gpu-shared --gpus=1 --constraint="gpu80" \
  --time=4:00:00 --account=YOUR_ACCOUNT
```
