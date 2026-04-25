"""
build_index.py — One-time RAG index builder for WE-Hybrid Terminal Agent

Run this ONCE on the EXPANSE login node (which has internet access).
The resulting ChromaDB index is stored on disk and reused every time you
run run_agent.py — no internet needed after this.

Usage:
    # On EXPANSE login node:
    python build_index.py

    # Point at your local repo files for extra context:
    python build_index.py --repo-dir /path/to/We_hybrid

    # Custom index output directory:
    python build_index.py --index-dir ~/westpa_index

    # Use a smaller/faster embedding model:
    python build_index.py --embed-model paraphrase-MiniLM-L3-v2

    # Rebuild from scratch (re-download + re-embed everything):
    python build_index.py --force

What it does:
    1. Loads the 11 hand-written EXPERT_DOCS from rag.py (always included)
    2. Fetches WESTPA documentation from GitHub (README, tutorials, analysis docs)
    3. Fetches PLUMED documentation excerpts
    4. Loads your local repo files (west.cfg, run.sh, md.in, etc.) if --repo-dir given
    5. Chunks long documents into ~512-token pieces
    6. Embeds everything with sentence-transformers (all-MiniLM-L6-v2, ~90MB)
    7. Stores in a persistent ChromaDB vector index on disk
"""

import argparse
import os
import sys
import re
import time
import textwrap
from pathlib import Path
from typing import List, Dict, Optional

# ── Optional pretty printing ──────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


def log(msg: str, style: str = ""):
    if RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Document chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    title: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict]:
    """
    Split a long text into overlapping chunks for embedding.
    Uses word-boundary splitting (not token-exact, but close enough for MiniLM).
    Returns list of dicts: {text, source, title}
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()
        if len(chunk_text_str) > 30:   # skip tiny fragments
            chunks.append({
                "text": chunk_text_str,
                "source": source,
                "title": title,
            })
        start += chunk_size - overlap

    return chunks


def chunk_by_section(
    text: str,
    source: str,
    title: str,
    max_chunk_words: int = 400,
) -> List[Dict]:
    """
    Split markdown/rst text at headings first, then by word count.
    Gives more coherent chunks than pure word-sliding.
    """
    # Split at markdown headings (# ## ### ===  --- etc.)
    section_pattern = re.compile(
        r"^(#{1,4}\s+.+|.+\n[=\-]{3,})$",
        re.MULTILINE,
    )
    positions = [m.start() for m in section_pattern.finditer(text)]
    positions.append(len(text))

    sections = []
    prev = 0
    for pos in positions:
        section = text[prev:pos].strip()
        if section:
            sections.append(section)
        prev = pos

    if not sections:
        sections = [text]

    chunks = []
    for section in sections:
        words = section.split()
        if len(words) <= max_chunk_words:
            if len(section.strip()) > 30:
                chunks.append({"text": section.strip(), "source": source, "title": title})
        else:
            # further split large sections
            sub_chunks = chunk_text(section, source, title,
                                    chunk_size=max_chunk_words, overlap=50)
            chunks.extend(sub_chunks)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# GitHub document fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch text from a URL. Returns None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "WE-Hybrid-RAG/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        log(f"  [yellow]⚠  Could not fetch {url}: {e}[/yellow]" if RICH
            else f"  Warning: Could not fetch {url}: {e}")
        return None


# WESTPA documentation URLs (GitHub raw content)
WESTPA_DOCS = [
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/README.rst",
        "title": "WESTPA README",
        "source": "westpa_github_readme",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/west_cfg.rst",
        "title": "WESTPA west.cfg Full Reference",
        "source": "westpa_west_cfg_ref",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/running.rst",
        "title": "WESTPA Running Simulations Guide",
        "source": "westpa_running_guide",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/analysis.rst",
        "title": "WESTPA Analysis Tools",
        "source": "westpa_analysis_guide",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/we_basics.rst",
        "title": "Weighted Ensemble Basics",
        "source": "westpa_we_basics",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/binning.rst",
        "title": "WESTPA Binning and Progress Coordinates",
        "source": "westpa_binning",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/hpc.rst",
        "title": "WESTPA HPC and ZMQ Distributed Execution",
        "source": "westpa_hpc_zmq",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/users_guide/bstates.rst",
        "title": "WESTPA Basis States and Initial States",
        "source": "westpa_bstates",
    },
    # Tutorials
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/tutorials/nacl_amber/README.rst",
        "title": "WESTPA NaCl AMBER Tutorial",
        "source": "westpa_tutorial_nacl",
    },
    {
        "url": "https://raw.githubusercontent.com/westpa/westpa/master/doc/tutorials/p53_amber/README.rst",
        "title": "WESTPA p53 AMBER Tutorial",
        "source": "westpa_tutorial_p53",
    },
]

# PLUMED documentation (relevant to ParMetaD)
PLUMED_DOCS = [
    {
        "url": "https://raw.githubusercontent.com/plumed/plumed2/master/CHANGES/v2.9.md",
        "title": "PLUMED v2.9 Changes",
        "source": "plumed_changelog",
    },
]

# ParMetaD / ParGaMD repository files to try fetching
REPO_GITHUB_DOCS = [
    {
        "url": "https://raw.githubusercontent.com/Sonti974948/ParMetaD/main/README.md",
        "title": "ParMetaD README",
        "source": "parmetad_readme",
    },
    {
        "url": "https://raw.githubusercontent.com/Sonti974948/ParGaMD/main/README.md",
        "title": "ParGaMD README",
        "source": "pargamd_readme",
    },
]


def fetch_all_web_docs() -> List[Dict]:
    """Fetch all remote documentation sources. Returns list of {text, source, title} dicts."""
    all_docs = []
    sources = WESTPA_DOCS + PLUMED_DOCS + REPO_GITHUB_DOCS

    log("\n[bold]Fetching web documentation...[/bold]" if RICH else "\nFetching web documentation...")

    for doc_meta in sources:
        url = doc_meta["url"]
        title = doc_meta["title"]
        source = doc_meta["source"]

        log(f"  Fetching: {title} ...", "dim" if RICH else "")
        content = fetch_url(url)
        if content and len(content.strip()) > 100:
            chunks = chunk_by_section(content, source=source, title=title)
            all_docs.extend(chunks)
            log(f"    ✓ {len(chunks)} chunks", "green" if RICH else "")
        else:
            log(f"    ✗ skipped (empty or unreachable)", "yellow" if RICH else "")
        time.sleep(0.3)  # be polite to GitHub

    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# Local repo file loader
# ─────────────────────────────────────────────────────────────────────────────

# File extensions to index from the local repo
INDEXABLE_EXTENSIONS = {
    ".py", ".sh", ".cfg", ".in", ".txt", ".md", ".rst", ".yaml", ".yml",
    ".plumed", ".dat", ".conf",
}

# Files/dirs to skip
SKIP_PATTERNS = {
    "__pycache__", ".git", "*.pyc", "node_modules", ".eggs",
    "dist", "build", "*.egg-info", "westpa_index", "chroma_db",
    "west.h5",   # binary HDF5
}


def should_skip(path: Path) -> bool:
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("*"):
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in path.parts or path.name == pattern:
            return True
    return False


def load_local_repo(repo_dir: str) -> List[Dict]:
    """
    Walk repo_dir and load all text files with indexable extensions.
    Returns list of {text, source, title} dicts ready for chunking.
    """
    repo_path = Path(repo_dir).resolve()
    if not repo_path.exists():
        log(f"[yellow]Warning: --repo-dir {repo_dir} not found, skipping.[/yellow]"
            if RICH else f"Warning: --repo-dir {repo_dir} not found, skipping.")
        return []

    log(f"\n[bold]Loading local repo files from: {repo_path}[/bold]"
        if RICH else f"\nLoading local repo files from: {repo_path}")

    all_chunks = []
    file_count = 0

    for fpath in sorted(repo_path.rglob("*")):
        if not fpath.is_file():
            continue
        if should_skip(fpath):
            continue
        if fpath.suffix.lower() not in INDEXABLE_EXTENSIONS:
            continue
        if fpath.stat().st_size > 500_000:   # skip huge files (> 500KB)
            continue

        try:
            content = fpath.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue

        if len(content) < 30:
            continue

        rel_path = fpath.relative_to(repo_path)
        source = f"local:{rel_path}"
        title = f"{fpath.name} ({rel_path.parent})"

        chunks = chunk_by_section(content, source=source, title=title, max_chunk_words=350)
        all_chunks.extend(chunks)
        file_count += 1

    log(f"  ✓ {file_count} files → {len(all_chunks)} chunks", "green" if RICH else "")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    index_dir: str,
    repo_dir: Optional[str],
    embed_model: str,
    force: bool,
    skip_web: bool,
):
    """Build the ChromaDB index from all document sources."""

    # ── Import RAG utilities ──────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from rag import index_documents, index_ready, get_expert_docs
    except ImportError as e:
        log(f"[red]Error importing rag.py: {e}[/red]" if RICH else f"Error importing rag.py: {e}")
        log("Make sure rag.py is in the same directory as build_index.py")
        sys.exit(1)

    # ── Check if already indexed ──────────────────────────────────────────────
    if not force and index_ready(index_dir):
        log(f"\n[green]✓ Index already exists at: {index_dir}[/green]"
            if RICH else f"\n✓ Index already exists at: {index_dir}")
        log("  Use --force to rebuild from scratch.")
        log("\n[bold]Index is ready for use by run_agent.py[/bold]"
            if RICH else "\nIndex is ready for use by run_agent.py")
        return

    log(f"\n[bold blue]Building WE-Hybrid RAG index → {index_dir}[/bold blue]"
        if RICH else f"\nBuilding WE-Hybrid RAG index → {index_dir}")

    all_docs: List[Dict] = []

    # ── 1. Expert hand-written docs ──────────────────────────────────────────
    log("\n[bold]Loading expert domain documents...[/bold]"
        if RICH else "\nLoading expert domain documents...")
    expert_docs = get_expert_docs()
    expert_chunks = []
    for doc in expert_docs:
        chunks = chunk_by_section(
            doc["text"],
            source=doc["source"],
            title=doc["title"],
            max_chunk_words=400,
        )
        expert_chunks.extend(chunks)
    all_docs.extend(expert_chunks)
    log(f"  ✓ {len(expert_docs)} expert docs → {len(expert_chunks)} chunks", "green" if RICH else "")

    # ── 2. Web documentation ─────────────────────────────────────────────────
    if not skip_web:
        web_chunks = fetch_all_web_docs()
        all_docs.extend(web_chunks)
        log(f"\n  ✓ Web docs total: {len(web_chunks)} chunks", "green" if RICH else "")
    else:
        log("\n[yellow]Skipping web fetch (--skip-web)[/yellow]"
            if RICH else "\nSkipping web fetch (--skip-web)")

    # ── 3. Local repo files ───────────────────────────────────────────────────
    if repo_dir:
        local_chunks = load_local_repo(repo_dir)
        all_docs.extend(local_chunks)
    else:
        # Try to auto-detect repo alongside we_wizard/terminal_agent/
        script_dir = Path(__file__).parent.resolve()
        candidates = [
            script_dir.parent.parent.parent,   # ParMetaD root (3 levels up)
            script_dir.parent.parent,           # we_wizard root
            Path.home() / "We_hybrid",
            Path.home() / "ParMetaD",
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.rglob("west.cfg")):
                log(f"\n[dim]Auto-detected repo at: {candidate}[/dim]"
                    if RICH else f"\nAuto-detected repo at: {candidate}")
                local_chunks = load_local_repo(str(candidate))
                all_docs.extend(local_chunks)
                break

    # ── 4. Deduplicate ────────────────────────────────────────────────────────
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc["text"][:200]   # compare first 200 chars
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    log(f"\n[bold]Total unique chunks to index: {len(unique_docs)}[/bold]"
        if RICH else f"\nTotal unique chunks to index: {len(unique_docs)}")

    # ── 5. Embed and store ────────────────────────────────────────────────────
    log(f"\n[bold]Embedding with model: {embed_model}[/bold]"
        if RICH else f"\nEmbedding with model: {embed_model}")
    log("  This takes 1-5 minutes depending on hardware. Run this on a GPU node for speed.")
    log("  (All computation stays local — no internet needed after model download)\n")

    try:
        index_documents(
            docs=unique_docs,
            index_dir=index_dir,
            model_name=embed_model,
            batch_size=64,
        )
    except Exception as e:
        log(f"[red]Indexing failed: {e}[/red]" if RICH else f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 6. Done ───────────────────────────────────────────────────────────────
    if RICH and console:
        console.print(Panel(
            f"[green bold]✅ Index built successfully![/green bold]\n\n"
            f"📁 Location: [cyan]{index_dir}[/cyan]\n"
            f"📊 Chunks indexed: [cyan]{len(unique_docs)}[/cyan]\n"
            f"🤖 Embedding model: [cyan]{embed_model}[/cyan]\n\n"
            f"[dim]The agent will automatically load this index.\n"
            f"Set [white]WESTPA_INDEX_DIR={index_dir}[/white] if you used a custom path.[/dim]",
            title="RAG Index Ready",
            border_style="green",
        ))
    else:
        print(f"\n{'='*60}")
        print(f"  ✅ Index built! Location: {index_dir}")
        print(f"  Chunks indexed: {len(unique_docs)}")
        print(f"{'='*60}")
        print(f"\nThe agent will automatically load this index.")
        print(f"Set WESTPA_INDEX_DIR={index_dir} if you used a custom path.")


# ─────────────────────────────────────────────────────────────────────────────
# Quick retrieval test
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval(index_dir: str, embed_model: str):
    """Run a few test queries to verify the index works."""
    sys.path.insert(0, os.path.dirname(__file__))
    from rag import retrieve

    test_queries = [
        "What is pcoord_len and how do I calculate it?",
        "How do I set up ZMQ distributed WESTPA on a cluster?",
        "What are basis states and initial states in WESTPA?",
        "How do I run w_ipa to compute free energy?",
        "What is the gamd-restart.dat file for?",
        "How do I use PLUMED with WESTPA for metadynamics?",
    ]

    log("\n[bold]Running retrieval test queries...[/bold]"
        if RICH else "\nRunning retrieval test queries...")

    for query in test_queries:
        results = retrieve(query, k=2, index_dir=index_dir, model_name=embed_model)
        if results:
            top = results[0]
            score_str = f"{top['score']:.3f}" if top.get('score') is not None else "N/A"
            log(f"\n  Query: [italic]{query}[/italic]" if RICH else f"\n  Query: {query}")
            log(f"  Best: [{score_str}] {top['title']} ({top['source']})")
            log(f"  Snippet: {top['text'][:120].strip()}...")
        else:
            log(f"\n  Query: {query}")
            log("  ✗ No results — index may be empty or corrupt", "red" if RICH else "")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB RAG index for WE-Hybrid Terminal Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Basic build (just expert docs + web fetch):
              python build_index.py

              # Include your local repo files:
              python build_index.py --repo-dir ~/We_hybrid

              # Custom index location:
              python build_index.py --index-dir /scratch/$USER/westpa_index

              # Build without internet (expert docs only):
              python build_index.py --skip-web

              # Force full rebuild:
              python build_index.py --force --repo-dir ~/We_hybrid

              # Test the index after building:
              python build_index.py --test-only
        """)
    )
    parser.add_argument(
        "--repo-dir", default=None,
        help="Path to your We_hybrid / ParMetaD+ParGaMD directory (optional, adds local files to index)"
    )
    parser.add_argument(
        "--index-dir", default=None,
        help="Output directory for the ChromaDB index (default: ~/.westpa_index)"
    )
    parser.add_argument(
        "--embed-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2, ~90MB)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the index from scratch even if it already exists"
    )
    parser.add_argument(
        "--skip-web", action="store_true",
        help="Skip fetching from GitHub (use only expert docs + local repo)"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Run retrieval test queries against an existing index (don't rebuild)"
    )
    parser.add_argument(
        "--no-test", action="store_true",
        help="Skip the retrieval test after building"
    )

    args = parser.parse_args()

    # Resolve index_dir
    if args.index_dir:
        index_dir = str(Path(args.index_dir).resolve())
    else:
        index_dir = str(Path.home() / ".westpa_index")

    embed_model = args.embed_model

    # Check dependencies
    try:
        import sentence_transformers  # noqa
        import chromadb               # noqa
    except ImportError as e:
        log(f"[red]Missing dependency: {e}[/red]" if RICH else f"Missing dependency: {e}")
        log("Install with:")
        log("  pip install sentence-transformers chromadb --break-system-packages")
        log("  # or: pip install sentence-transformers chromadb --user")
        sys.exit(1)

    if args.test_only:
        test_retrieval(index_dir, embed_model)
        return

    # Build the index
    build_index(
        index_dir=index_dir,
        repo_dir=args.repo_dir,
        embed_model=embed_model,
        force=args.force,
        skip_web=args.skip_web,
    )

    # Run a quick test unless --no-test
    if not args.no_test:
        test_retrieval(index_dir, embed_model)


if __name__ == "__main__":
    main()
