"""
llm_agent.py
LangGraph + HuggingFace Transformers agent for WE-Hybrid simulation setup.

Architecture:
  User message → LangGraph chatbot node → Llama (local, HuggingFace) → tool calls → config update
  Falls back to rule-based agent if GPU/model not available.

Requires (pip install on EXPANSE login node):
  pip install transformers accelerate langchain-huggingface langchain-core langgraph bitsandbytes

Model setup (run on EXPANSE login node — has internet):
  huggingface-cli login          # paste your HF token once
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

  # No HF account? Use ungated model instead:
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct

RAG (optional but recommended):
  pip install sentence-transformers chromadb
  python build_index.py --repo-dir /path/to/We_hybrid   (run once on login node)
"""

import json
import sys
import os
import logging
from pathlib import Path
from typing import Annotated, TypedDict

# Suppress verbose model-loading output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*inputs will be cast.*")

# LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# LangChain core
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# HuggingFace backend
try:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, pipeline as hf_pipeline,
    )
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    HF_AVAILABLE = True
except ImportError as _hf_err:
    HF_AVAILABLE = False
    _hf_import_error = str(_hf_err)

# Local
from config_schema import SimConfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generators import generate_zip, compute_pcoord_len


# ─────────────────────────────────────────────────────────────────────────────
# RAG — optional; gracefully disabled if not installed / index not built
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_index_dir() -> str:
    """
    Find the ChromaDB index directory in order of preference:
      1. WESTPA_INDEX_DIR environment variable
      2. westpa_index/ sibling of this script (placed by build_index.py default)
      3. ~/.westpa_index (home dir, build_index.py default)
    """
    env_dir = os.environ.get("WESTPA_INDEX_DIR")
    if env_dir and Path(env_dir).exists():
        return env_dir

    script_dir = Path(__file__).parent.resolve()
    local_dir = script_dir / "westpa_index"
    if local_dir.exists():
        return str(local_dir)

    home_dir = Path.home() / ".westpa_index"
    if home_dir.exists():
        return str(home_dir)

    # return default even if it doesn't exist yet — rag.index_ready() will
    # return False and the tool will gracefully report no index
    return str(home_dir)


try:
    from rag import retrieve, format_retrieved_context, index_ready
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

_RAG_INDEX_DIR = _resolve_index_dir()
_RAG_EMBED_MODEL = os.environ.get("WESTPA_EMBED_MODEL", "all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────────────────────
# Shared config state (mutable, accessed by tools and main loop)
# ─────────────────────────────────────────────────────────────────────────────

_sim_config = SimConfig()


def get_config() -> SimConfig:
    return _sim_config


def reset_config():
    global _sim_config
    _sim_config = SimConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Domain system prompt — injected as context into every LLM call
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for setting up WE-Hybrid biased molecular dynamics simulations.
You help researchers configure ParMetaD (WESTPA + PLUMED Metadynamics) and ParGaMD (WESTPA + Gaussian Accelerated MD) simulations.

## Your Role
Guide the user through collecting all required simulation parameters, then generate ready-to-submit HPC files.
Be conversational and helpful. Answer scientific questions inline. Extract multiple values from a single message when possible.

## Domain Knowledge

### Methods
- **ParMetaD**: Combines WESTPA weighted ensemble with PLUMED metadynamics. Each walker runs metadynamics with its own HILLS file. Supported backends: AMBER (pmemd.cuda) or OpenMM.
- **ParGaMD**: Combines WESTPA with Gaussian Accelerated MD. Requires a cGaMD pre-run to get gamd-restart.dat before starting the WE simulation. Only AMBER backend.

### Critical Constraint (ALWAYS enforce this)
`pcoord_len = nstlim / ntpr + 1`
- nstlim: number of MD steps per WE segment
- ntpr: output frequency in steps
- pcoord_len: number of progress coordinate data points (auto-set in west.cfg)
- Example: nstlim=50000, ntpr=500 → pcoord_len=101
- nstlim MUST be divisible by ntpr. If not, ask the user to adjust.

### Key Parameters
- **NODELOC / scratch_dir**: Fast parallel filesystem for per-walker temp files (e.g. /expanse/lustre/scratch/username/project)
- **pcoord (progress coordinate)**: Observable WESTPA uses to bin walkers. Common choices: RMSD from reference, radius of gyration, inter-residue distance
- **bins**: Partition the pcoord space. Each bin maintains a target number of walkers. Typical: 20-50 bins spanning full CV range
- **walkers per bin**: More = better statistics, more GPU time. Typical: 2-8
- **pcoord_ndim**: 1D (single CV) or 2D (two CVs for richer FES)
- **sigma0D / sigma0P**: GaMD boost thresholds (kcal/mol). 6.0/6.0 is standard for proteins in implicit solvent
- **iE=2**: Upper bound GaMD (recommended for proteins), iE=1 = lower bound
- **HILLS file**: PLUMED output recording Gaussian bias. Copied from parent to child walker each WE iteration
- **COLVAR**: PLUMED output with CV values — this feeds back to WESTPA as the pcoord
- **ZMQ**: ZeroMQ — how WESTPA distributes work across nodes. Master process + SSH workers on compute nodes

### HPC / SLURM
- ParGaMD workflow: sbatch cMD/run_cmd.sh → get JOBID → sbatch --dependency=afterok:JOBID run_WE.sh
- Typical walltime for WE run: 24-48 hours (checkpoints every iteration, restartable)
- SLURM account: billing project code (e.g. ucd192)

### File Structure Generated
- env.sh: environment setup (modules, paths, conda)
- west.cfg: WESTPA config (pcoord, bins, walkers, pcoord_len)
- run_WE.sh: SLURM submission script
- node.sh: per-node ZMQ worker
- westpa_scripts/runseg.sh: per-walker MD runner
- common_files/plumed.dat: PLUMED metadynamics config (ParMetaD)
- common_files/md.in: AMBER MD input
- cMD/run_cmd.sh: cGaMD pre-run script (ParGaMD only)

## Required Information to Collect
Collect ALL of these before generating files:
1. method (parmetad or parGaMD)
2. backend (amber or openmm — parGaMD is always amber)
3. hpc.scratch_dir (absolute path on cluster)
4. hpc.conda_env (conda environment with WESTPA)
5. hpc.account (SLURM billing account)
6. hpc.partition (SLURM partition, e.g. gpu)
7. hpc.nodes and hpc.gpus_per_node
8. hpc.walltime (HH:MM:SS)
9. paths.amberhome (if amber backend)
10. paths.plumed_kernel_dir (if parmetad)
11. md.temperature (Kelvin)
12. md.timestep (ps, typically 0.002)
13. md.nsteps (nstlim — steps per segment)
14. md.ntpr (output frequency, must divide nsteps)
15. westpa.pcoord_ndim (1 or 2)
16. westpa.bin_boundaries (list of boundary values per dimension)
17. westpa.bin_target_counts (walkers per bin)
18. westpa.max_iterations
19. gamd.sigma0D, gamd.sigma0P, gamd.iE (parGaMD only)

## Output Format — ALWAYS follow this exactly

Every reply MUST start with a <params> block (even if empty):

<params>{"field": value, ...}</params>
Your conversational response here.

Rules for the <params> block:
- Include ONLY fields the user explicitly mentioned in this message
- Use exact dot-notation keys from "Required Information to Collect"
- If nothing new was mentioned, output: <params>{}</params>
- Strings in double quotes, numbers unquoted, lists as JSON arrays
- NEVER skip the <params> block — it is required every turn

Examples:
  User: "Method is ParMetaD, backend AMBER"
  → <params>{"method": "parmetad", "backend": "amber"}</params>
  Got it — ParMetaD with AMBER. Next I need your scratch directory...

  User: "4 nodes, 4 GPUs, account ucd192"
  → <params>{"hpc.nodes": 4, "hpc.gpus_per_node": 4, "hpc.account": "ucd192"}</params>
  Perfect. What SLURM partition should I use?

  User: "What is pcoord_len?"
  → <params>{}</params>
  pcoord_len is the number of progress coordinate points per WE iteration...

## Behavior Rules
- Extract all values from the user's message in ONE <params> block
- After the <params> block, write a helpful conversational response
- When config is complete, tell the user to type 'generate'
- Be concise — this user is a PhD researcher, skip basic explanations unless asked
- For "what is X" questions, answer directly and scientifically
"""


# ─────────────────────────────────────────────────────────────────────────────
# Config update — called by chatbot_node after parsing LLM <params> block
# ─────────────────────────────────────────────────────────────────────────────

def apply_config_updates(updates: dict) -> list:
    """Apply a dict of {dot-notation-key: value} updates to _sim_config. Returns log lines."""
    global _sim_config
    log = []
    for key, value in updates.items():
        parts = key.split(".", 1)
        try:
            if len(parts) == 1:
                setattr(_sim_config, key, value)
                log.append(f"✓ {key} = {value}")
            else:
                section, field = parts
                sub = getattr(_sim_config, section, None)
                if sub is None:
                    log.append(f"✗ unknown section: {section}")
                    continue
                setattr(sub, field, value)
                log.append(f"✓ {section}.{field} = {value}")
        except Exception as e:
            log.append(f"✗ {key}: {e}")

    # Recompute pcoord_len whenever md steps change
    if any("nsteps" in k or "ntpr" in k for k in updates):
        if _sim_config.md.nsteps > 0 and _sim_config.md.ntpr > 0:
            _sim_config.md.pcoord_len = compute_pcoord_len(
                _sim_config.md.nsteps, _sim_config.md.ntpr
            )
            log.append(f"↳ pcoord_len = {_sim_config.md.pcoord_len}")
    return log


# ─────────────────────────────────────────────────────────────────────────────
# RAG context injection — called before LLM when user asks a question
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_WORDS = {"what", "how", "why", "when", "where", "which", "explain",
                   "describe", "tell", "difference", "mean", "means", "define"}

def _looks_like_question(text: str) -> bool:
    words = set(text.lower().split())
    return "?" in text or bool(words & _QUESTION_WORDS)


def _rag_context_for(query: str) -> str:
    """Return RAG context string to prepend to system prompt, or '' if unavailable."""
    if not _RAG_AVAILABLE or not index_ready(_RAG_INDEX_DIR):
        return ""
    try:
        results = retrieve(query=query, k=3,
                           index_dir=_RAG_INDEX_DIR, model_name=_RAG_EMBED_MODEL)
        return format_retrieved_context(results) if results else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# GPU / hardware check
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu() -> dict:
    """Return GPU availability info."""
    if not HF_AVAILABLE:
        return {
            "cuda_available": False,
            "device": "cpu",
            "gpu_name": None,
            "vram_gb": 0,
            "message": f"HuggingFace/torch not installed: {_hf_import_error if not HF_AVAILABLE else ''}",
        }
    cuda = torch.cuda.is_available()
    if cuda:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return {
            "cuda_available": True,
            "device": f"cuda:{idx}",
            "gpu_name": name,
            "vram_gb": round(vram, 1),
            "message": f"{name} ({round(vram,1)} GB VRAM)",
        }
    return {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": 0,
        "message": "No CUDA GPU detected — model will run on CPU (very slow, not recommended)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_model(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    quantization: str = "8bit",
    hf_token: str = None,
) -> "ChatHuggingFace":
    """
    Load a HuggingFace causal LM and wrap it for LangGraph tool calling.

    model_id:     Any HuggingFace model ID. Llama 3.x recommended for tool use.
    quantization: '4bit' (~4GB VRAM), '8bit' (~8GB VRAM), 'fp16' (~16GB VRAM)
    hf_token:     HuggingFace access token (needed for gated models like Llama).
                  Falls back to HF_TOKEN env var.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace packages not installed.\n"
            "Run: pip install transformers accelerate langchain-huggingface bitsandbytes"
        )

    token = hf_token or os.environ.get("HF_TOKEN")

    # ── Quantization config ───────────────────────────────────────────────────
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        dtype = None
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None
    else:  # fp16
        bnb_config = None
        dtype = torch.float16

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    model.eval()

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    hf_pipe = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=hf_pipe, verbose=False)


def rag_status() -> dict:
    """
    Return RAG availability info for display in the UI.
    Returns dict with keys: available (bool), index_ready (bool), index_dir (str), message (str)
    """
    if not _RAG_AVAILABLE:
        return {
            "available": False,
            "index_ready": False,
            "index_dir": _RAG_INDEX_DIR,
            "message": "RAG libraries not installed (pip install sentence-transformers chromadb)",
        }
    ready = index_ready(_RAG_INDEX_DIR)
    return {
        "available": True,
        "index_ready": ready,
        "index_dir": _RAG_INDEX_DIR,
        "message": (
            f"RAG index loaded from {_RAG_INDEX_DIR}"
            if ready
            else f"RAG libraries present but no index at {_RAG_INDEX_DIR} — run build_index.py"
        ),
    }


def build_graph(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    quantization: str = "8bit",
    hf_token: str = None,
):
    """
    Build and compile the LangGraph agent using a local HuggingFace model.

    Architecture (no tool calling — more reliable with local models):
      User message
        → optionally inject RAG context into system prompt
        → LLM generates response with <params>{...}</params> block
        → parse <params> block and apply config updates directly
        → return cleaned response text to user
    """
    import re as _re

    llm = load_hf_model(model_id=model_id, quantization=quantization, hf_token=hf_token)

    def chatbot_node(state: AgentState):
        # Get the latest user message for RAG lookup
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user = user_msgs[-1].content if user_msgs else ""

        # Optionally inject RAG context
        system = SYSTEM_PROMPT
        if _looks_like_question(last_user):
            rag_ctx = _rag_context_for(last_user)
            if rag_ctx:
                system = SYSTEM_PROMPT + f"\n\n## Relevant Documentation\n{rag_ctx}"

        messages = [SystemMessage(content=system)] + state["messages"]
        response = llm.invoke(messages)

        content = response.content if isinstance(response.content, str) else str(response.content)

        # ── Parse <params> block and apply config updates ─────────────────────
        params_match = _re.search(r"<params>\s*(\{.*?\})\s*</params>",
                                  content, _re.DOTALL)
        if params_match:
            try:
                updates = json.loads(params_match.group(1))
                if updates:
                    apply_config_updates(updates)
            except (json.JSONDecodeError, Exception):
                pass  # malformed JSON — skip silently
            # Strip the <params> block from the displayed response
            content = _re.sub(r"<params>\s*\{.*?\}\s*</params>\s*",
                               "", content, flags=_re.DOTALL).strip()

        return {"messages": [AIMessage(content=content)]}

    graph = StateGraph(AgentState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Generate files from current config
# ─────────────────────────────────────────────────────────────────────────────

def generate_files_from_config(output_path: str) -> tuple[bool, str]:
    """
    Generate simulation files from the current config and save to output_path.
    Returns (success, message).
    """
    complete, missing = _sim_config.is_complete()
    if not complete:
        return False, f"Config incomplete. Missing: {', '.join(missing)}"

    try:
        config_dict = _sim_config.to_dict()
        zip_buf = generate_zip(config_dict)

        import zipfile
        with zipfile.ZipFile(zip_buf) as zf:
            zf.extractall(output_path)

        return True, f"Files extracted to: {output_path}"
    except Exception as e:
        return False, f"Generation failed: {e}"
