"""
run_agent.py — WE-Hybrid Terminal Agent
Powered by LangGraph + HuggingFace Transformers (Llama 3.1) — fully local, no API key.

Usage on GPU node:
    python run_agent.py
    python run_agent.py --model meta-llama/Llama-3.1-8B-Instruct  # default
    python run_agent.py --model Qwen/Qwen2.5-7B-Instruct          # no HF token needed
    python run_agent.py --model meta-llama/Llama-3.2-3B-Instruct  # faster, smaller
    python run_agent.py --quantization 4bit                        # less VRAM (~4GB)
    python run_agent.py --quantization fp16                        # full precision (~16GB)
    python run_agent.py --fallback                                 # rule-based, no GPU needed
    python run_agent.py --output ./my_sim                          # set output directory

HuggingFace token (needed for gated models like Llama):
    export HF_TOKEN=hf_xxxxxxxxxxxx
    # or pass --hf-token hf_xxxxxxxxxxxx
    # Get token at: https://huggingface.co/settings/tokens
    # Accept Llama license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Rich terminal UI ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.columns import Columns
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── LangGraph agent ───────────────────────────────────────────────────────────
try:
    from llm_agent import (
        build_graph, check_gpu,
        generate_files_from_config, get_config, reset_config, TOOLS,
        rag_status,
    )
    from config_schema import SimConfig
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    _import_error = str(e)
    def rag_status():
        return {"available": False, "index_ready": False, "message": "LangGraph not installed"}
    def check_gpu():
        return {"cuda_available": False, "message": "packages not installed"}

# ── Fallback rule-based agent ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from agent import ChatAgent as FallbackAgent
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────────────────────────────────────────────────────────
# Console setup
# ─────────────────────────────────────────────────────────────────────────────

console = Console() if RICH_AVAILABLE else None


def print_md(text: str):
    """Render markdown in terminal (Rich) or plain text fallback."""
    if RICH_AVAILABLE and console:
        console.print(Markdown(text))
    else:
        print(text)


def print_panel(text: str, title: str = "", style: str = "blue"):
    if RICH_AVAILABLE and console:
        console.print(Panel(Markdown(text), title=title, border_style=style, padding=(0, 1)))
    else:
        print(f"\n{'='*60}")
        if title:
            print(f"  {title}")
        print(text)
        print('='*60)


def print_rule(title: str = ""):
    if RICH_AVAILABLE and console:
        console.print(Rule(title, style="dim"))
    else:
        print(f"\n{'─'*60} {title}")


def print_config_sidebar(config: SimConfig):
    """Render the live config panel."""
    if not RICH_AVAILABLE or not console:
        return

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    table.add_column("Key", style="dim", width=18)
    table.add_column("Value", style="bold")

    complete, missing = config.is_complete()
    for key, val in config.summary_table():
        style = "green" if val not in ("—", "0", "") else "dim"
        table.add_row(key, Text(val, style=style))

    status = "✅ Ready to generate" if complete else f"⏳ {len(missing)} fields remaining"
    status_style = "green" if complete else "yellow"

    console.print(Panel(
        table,
        title="[bold]⚙ Current Config[/bold]",
        subtitle=f"[{status_style}]{status}[/{status_style}]",
        border_style="blue",
        padding=(0, 1),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
# ⚗️  WE-Hybrid Terminal Agent
**Powered by LangGraph + HuggingFace (Llama 3.1) — fully local, no API key**

Setup a complete **ParMetaD** or **ParGaMD** simulation directory through conversation.
Type naturally — I'll extract your parameters and ask for what's missing.

**Tips:**
- Ask *"what is pcoord_len?"* at any point for explanations
- Say multiple values at once: *"4 nodes, 4 GPUs, partition gpu"*
- Type `show config` to see current settings
- Type `generate` when ready to create your files
- Type `quit` to exit
"""


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph chat loop
# ─────────────────────────────────────────────────────────────────────────────

def run_langgraph_agent(model: str, quantization: str, hf_token: str, output_dir: str):
    """Main loop using LangGraph + HuggingFace Transformers."""
    console.print(Panel(Markdown(BANNER), border_style="bright_blue", padding=(1, 2)))

    # GPU check
    gpu = check_gpu()
    if gpu["cuda_available"]:
        console.print(f"[green]✓ GPU: {gpu['message']}[/green]")
    else:
        console.print(f"[yellow]⚠  {gpu['message']}[/yellow]")
        console.print("[dim]  For real use, run on a GPU node: sinteractive --partition=gpu-shared --gpus=1[/dim]")

    # RAG status
    rs = rag_status()
    if rs["index_ready"]:
        console.print(f"[green]✓ RAG index loaded — deep WESTPA Q&A enabled[/green]")
    elif rs["available"]:
        console.print(
            f"[yellow]⚠  RAG index not found[/yellow] — "
            f"[dim]run: python build_index.py --repo-dir ~/We_hybrid[/dim]"
        )
    else:
        console.print(f"[dim]ℹ  RAG not enabled — pip install sentence-transformers chromadb to add it[/dim]")

    # Load model (this is the slow step — downloading weights on first run)
    console.print(f"\n[bold blue]Loading model: {model} ({quantization})[/bold blue]")
    console.print("[dim]  First run: downloads weights to ~/.cache/huggingface/ (~5-15GB)[/dim]")
    console.print("[dim]  Subsequent runs: loads from cache in ~30-60 seconds[/dim]\n")

    try:
        with console.status(
            f"[bold blue]Loading {model.split('/')[-1]} into GPU...[/bold blue]",
            spinner="dots"
        ):
            graph = build_graph(
                model_id=model,
                quantization=quantization,
                hf_token=hf_token,
            )
    except Exception as e:
        err = str(e)
        console.print(f"[red]✗ Failed to load model: {err}[/red]")
        if "gated" in err.lower() or "token" in err.lower() or "401" in err:
            console.print(
                "\n[yellow]This model requires a HuggingFace account and accepted license.[/yellow]\n"
                "  1. Create account at https://huggingface.co\n"
                "  2. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct\n"
                "  3. Get token at https://huggingface.co/settings/tokens\n"
                "  4. Set: export HF_TOKEN=hf_xxxxxxxxxxxx\n\n"
                "[dim]Or use a model that needs no token:[/dim]\n"
                "  python run_agent.py --model Qwen/Qwen2.5-7B-Instruct"
            )
        if FALLBACK_AVAILABLE:
            console.print("\n[dim]Falling back to rule-based agent...[/dim]\n")
            run_fallback_agent(output_dir)
        return

    console.print(f"[green]✓ Model loaded![/green]\n")

    # Initial greeting
    graph_state = {"messages": []}
    greeting = (
        "Hello! I'm your WE-Hybrid simulation setup assistant, powered by Llama running locally on your GPU. "
        "I'll help you generate all the files needed for a ParMetaD or ParGaMD simulation.\n\n"
        "Let's start: **which method do you want to set up?**\n"
        "- **ParMetaD** — WESTPA + PLUMED Metadynamics (AMBER or OpenMM)\n"
        "- **ParGaMD** — WESTPA + Gaussian Accelerated MD (AMBER, requires cGaMD pre-run)"
    )
    print_panel(greeting, title="🤖 Agent", style="bright_blue")
    print_config_sidebar(get_config())

    while True:
        print_rule()

        # User input
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            else:
                user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/dim]")
            break

        if not user_input:
            continue

        # Special commands
        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! Your config is saved in memory — restart to continue.[/dim]")
            break

        if cmd in ("show config", "config", "status"):
            print_config_sidebar(get_config())
            continue

        if cmd in ("generate", "yes", "go", "done", "create files"):
            complete, missing = get_config().is_complete()
            if not complete:
                console.print(f"[yellow]Config not yet complete. Still need: {', '.join(missing)}[/yellow]")
                continue
            _do_generate(output_dir)
            break

        if cmd in ("reset", "restart"):
            reset_config()
            console.print("[yellow]Config reset. Starting over...[/yellow]")
            graph_state = {"messages": []}
            continue

        # Normal LLM turn
        graph_state["messages"].append(HumanMessage(content=user_input))

        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            try:
                result = graph.invoke(graph_state)
                graph_state = result
            except Exception as e:
                console.print(f"[red]Agent error: {e}[/red]")
                continue

        # Extract and display the last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            last_ai = ai_messages[-1]
            reply = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
            if reply.strip():
                print_panel(reply, title="🤖 Agent", style="bright_blue")

        # Refresh config panel after every turn
        print_config_sidebar(get_config())

        # Check if complete after LLM turn
        complete, _ = get_config().is_complete()
        if complete:
            console.print(
                "\n[bold green]✅ Config complete! Type [white]generate[/white] to create your simulation files.[/bold green]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fallback rule-based loop (if Ollama not available)
# ─────────────────────────────────────────────────────────────────────────────

def run_fallback_agent(output_dir: str):
    """Rule-based agent fallback — same UX but no LLM."""
    if not FALLBACK_AVAILABLE:
        print("Error: neither Ollama nor fallback agent available.")
        sys.exit(1)

    agent = FallbackAgent()

    print_panel(
        "⚠️ Running in **fallback mode** (rule-based, no LLM).\n"
        "Install Ollama and pull a model for the full AI-powered experience.\n\n"
        + BANNER,
        title="WE-Hybrid Setup Agent (Fallback Mode)",
        style="yellow",
    )

    # Start the agent
    result = agent.process("")
    print_panel(result["reply"], title="🤖 Agent", style="yellow")

    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            else:
                user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = agent.process(user_input)
        print_panel(result["reply"], title="🤖 Agent", style="yellow")

        if result.get("ready_to_generate"):
            _do_generate_from_dict(result["config"], output_dir)
            break


# ─────────────────────────────────────────────────────────────────────────────
# File generation
# ─────────────────────────────────────────────────────────────────────────────

def _do_generate(output_dir: str):
    """Generate files from LangGraph agent's config."""
    os.makedirs(output_dir, exist_ok=True)

    if RICH_AVAILABLE:
        with console.status(f"[bold green]Generating files into {output_dir}...[/bold green]", spinner="dots"):
            success, msg = generate_files_from_config(output_dir)
    else:
        print(f"Generating files into {output_dir}...")
        success, msg = generate_files_from_config(output_dir)

    if success:
        print_panel(
            f"🎉 **Simulation files generated!**\n\n"
            f"📁 Output: `{output_dir}`\n\n"
            f"**Next steps:**\n"
            f"1. Copy your system files (`.prmtop`, `.pdb`, `.rst`) into `{output_dir}/common_files/`\n"
            f"2. Copy basis state restart to `{output_dir}/bstates/bstate.rst`\n"
            f"3. Read `{output_dir}/SETUP_INSTRUCTIONS.md` for exact `sbatch` commands\n\n"
            f"```bash\ncd {output_dir}\ncat SETUP_INSTRUCTIONS.md\n```",
            title="✅ Done",
            style="green",
        )
    else:
        print_panel(f"❌ Generation failed:\n{msg}", title="Error", style="red")


def _do_generate_from_dict(config_dict: dict, output_dir: str):
    """Generate files from fallback agent's config dict."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from generators import generate_zip
    import zipfile

    os.makedirs(output_dir, exist_ok=True)
    try:
        buf = generate_zip(config_dict)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(output_dir)
        print_panel(
            f"🎉 **Files generated!** See `{output_dir}/SETUP_INSTRUCTIONS.md`",
            title="✅ Done", style="green"
        )
    except Exception as e:
        print_panel(f"❌ Failed: {e}", title="Error", style="red")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WE-Hybrid Terminal Agent — LangGraph + HuggingFace (Llama 3.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (--model):
  meta-llama/Llama-3.1-8B-Instruct   best quality, needs HF token  [default]
  meta-llama/Llama-3.2-3B-Instruct   faster, smaller, needs HF token
  Qwen/Qwen2.5-7B-Instruct           no HF token needed, excellent quality
  Qwen/Qwen2.5-3B-Instruct           no HF token needed, fast

HF token setup (for Llama models):
  1. https://huggingface.co/settings/tokens  → create token
  2. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  → accept license
  3. export HF_TOKEN=hf_xxxxxxxxxxxx
        """
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--quantization", default="8bit", choices=["4bit", "8bit", "fp16"],
        help="Model quantization: 4bit (~4GB VRAM), 8bit (~8GB VRAM), fp16 (~16GB VRAM). Default: 8bit"
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output", default="./we_simulation",
        help="Output directory for generated files (default: ./we_simulation)"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Force rule-based fallback agent (no GPU required)"
    )
    args = parser.parse_args()

    if not RICH_AVAILABLE:
        print("Warning: 'rich' not installed. Install with: pip install rich")
        print("Continuing with plain text output...\n")

    if args.fallback or not LANGGRAPH_AVAILABLE:
        if not LANGGRAPH_AVAILABLE and not args.fallback:
            print(f"LangGraph/HuggingFace packages not available ({_import_error})")
            print("Falling back to rule-based agent.")
            print("To enable LLM: pip install transformers accelerate langchain-huggingface langgraph\n")
        run_fallback_agent(args.output)
    else:
        run_langgraph_agent(
            model=args.model,
            quantization=args.quantization,
            hf_token=args.hf_token,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
