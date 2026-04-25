# WE-Hybrid Agent

AI-assisted setup tools for **ParMetaD** and **ParGaMD** weighted ensemble biased MD simulations on HPC clusters (EXPANSE, etc.).

Two independent approaches live in separate branches — pick whichever fits your workflow:

| Branch | Approach | Requires | Best for |
|--------|----------|----------|----------|
| [`hf-agent`](../../tree/hf-agent) | HuggingFace LangGraph terminal agent | GPU on cluster (V100+) | Self-contained, no external API |
| [`mcp-server`](../../tree/mcp-server) | Claude + MCP server | Claude API token (any tier) | Claude Code, Cursor, Antigravity |

---

## What does it do?

Both tools guide you through setting up a complete ParMetaD or ParGaMD simulation:

- Download and prepare protein structures (PDB → AMBER topology via tleap)
- Generate all WESTPA input files: `west.cfg`, `runseg.sh`, `run_WE.sh`, `node.sh`, `env.sh`, `plumed.dat`, `md.in`
- Validate critical constraints (pcoord_len, nsteps % ntpr, bin boundaries)
- Answer questions about WESTPA, PLUMED, GaMD, and AMBER
- Debug failed runs from log files

---

## Quick start

```bash
# Clone the repo
git clone https://github.com/Sonti974948/WE-hybrid-agent.git

# Switch to your preferred branch
git checkout hf-agent      # HuggingFace terminal agent
# or
git checkout mcp-server    # Claude + MCP server
```

Then follow the `README.md` in that branch.

---

## Background

**ParMetaD** = WESTPA + PLUMED well-tempered metadynamics  
**ParGaMD** = WESTPA + Gaussian Accelerated MD  

Both methods use the Weighted Ensemble (WE) strategy to sample rare events (e.g. protein folding) orders of magnitude faster than plain MD.

See [WESTPA](https://westpa.github.io/westpa/) and [PLUMED](https://www.plumed.org/) for full documentation.
