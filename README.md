# WE-Hybrid Agent — Claude + MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives any compatible AI coding tool direct tools to set up **ParMetaD** and **ParGaMD** weighted ensemble simulations — including SSH remote execution on EXPANSE with no manual file transfers.

Works with **Claude Code**, **Cursor**, and **Google Antigravity** out of the box.

---

## Overview

```
mcp_server.py          # MCP server — registers tools with your AI IDE
CLAUDE.md              # Full domain knowledge (WESTPA, PLUMED, GaMD, AMBER)
generators.py          # Generates all WE simulation input files
requirements.txt       # Python dependencies (mcp, paramiko, requests)
tools/
  executor.py          # SSH ControlMaster + local execution abstraction
  file_generator.py    # Wraps generators.py, writes files locally or via SFTP
  structure_prep.py    # PDB fetch + tleap system preparation
```

### MCP Tools exposed

| Tool | What it does |
|------|-------------|
| `connect_to_cluster` | SSH into EXPANSE (or any cluster) via ControlMaster — one TOTP, then silent |
| `disconnect_from_cluster` | Close SSH session |
| `get_server_mode` | Show current mode (SSH / local / laptop) and available binaries |
| `fetch_pdb_structure` | Download PDB from RCSB by 4-letter code |
| `prepare_amber_system` | Run tleap to build topology — locally or on cluster via SSH |
| `validate_simulation_config` | Catch parameter errors before generating files |
| `generate_simulation_files` | Write all WESTPA input files — locally or to cluster scratch via SFTP |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Local (laptop / WSL) |
| Claude API token | Any tier. Get one at [console.anthropic.com](https://console.anthropic.com) |
| Compatible AI IDE | Claude Code, Cursor, or Google Antigravity (see setup below) |
| SSH key for EXPANSE | For key-based auth (bypasses DUO for non-interactive sessions) |

---

## Installation (once)

```bash
git clone https://github.com/Sonti974948/WE-hybrid-agent.git
cd WE-hybrid-agent
git checkout mcp-server

pip install -r requirements.txt
```

---

## SSH Key Setup for EXPANSE (one-time, highly recommended)

EXPANSE uses DUO/TOTP. Set up a key so the MCP server can connect without prompting:

```bash
# 1. Generate a dedicated key on your laptop / WSL
ssh-keygen -t ed25519 -f ~/.ssh/id_expanse

# 2. Copy public key to EXPANSE (password + TOTP required once)
ssh-copy-id -i ~/.ssh/id_expanse.pub ssonti@login.expanse.sdsc.edu

# 3. Add ControlMaster config to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'

Host expanse
    HostName login.expanse.sdsc.edu
    User ssonti
    IdentityFile ~/.ssh/id_expanse
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
EOF

# 4. Test (should NOT ask for TOTP)
ssh expanse "echo connected"
```

Once ControlMaster is set up, open one terminal with `ssh expanse` and leave it running. That terminal holds the socket — everything else (the MCP server, your IDE) reuses it silently.

---

## Setup by IDE

### Claude Code

```bash
# Register the MCP server (run once)
claude mcp add we-hybrid python /full/path/to/WE-hybrid-agent/mcp_server.py

# Verify
claude mcp list   # should show ✓ we-hybrid

# Start a session
claude
```

> **Windows/WSL:** use the WSL path, e.g.  
> `claude mcp add we-hybrid python /mnt/c/Users/you/WE-hybrid-agent/mcp_server.py`

---

### Cursor

1. Open or create `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "we-hybrid": {
      "command": "python",
      "args": ["/full/path/to/WE-hybrid-agent/mcp_server.py"]
    }
  }
}
```

2. In Cursor: **Settings → Features → MCP** — enable MCP and restart.

3. The tools appear automatically in Cursor's agent panel. Ask Cursor:  
   *"Connect to EXPANSE and set up a ParMetaD simulation for Chignolin"*

> **Windows:** use the full Windows path, e.g.  
> `C:\\Users\\you\\WE-hybrid-agent\\mcp_server.py`  
> **WSL from Cursor:** use the WSL path:  
> `wsl python /mnt/c/Users/you/WE-hybrid-agent/mcp_server.py`

---

### Google Antigravity

1. Open or create `~/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "we-hybrid": {
      "command": "python",
      "args": ["/full/path/to/WE-hybrid-agent/mcp_server.py"]
    }
  }
}
```

2. Restart Antigravity. The tools are available in the agent panel immediately.

3. In the Antigravity chat, ask:  
   *"Use the we-hybrid tools to set up a ParMetaD simulation for Chignolin on EXPANSE"*

> Antigravity's agent will call the tools autonomously — you just answer its follow-up questions.

---

## Typical Workflow (all three IDEs)

Once registered, the session looks identical regardless of which IDE you're using:

### Step 1 — Open the ControlMaster terminal

```bash
# In a separate terminal (keep open):
ssh expanse
```

Enter TOTP once. This stays alive for 8 hours.

### Step 2 — In your AI IDE, start the workflow

Tell your AI assistant:

> *"Set up a ParMetaD simulation for Chignolin (1UAO) on EXPANSE. My scratch directory is /expanse/lustre/scratch/ssonti/chignolin, account is ucd192, 4 nodes × 4 GPUs."*

The assistant will call tools in this order:

```
1. connect_to_cluster(host="expanse", use_control_master=true,
                      setup_cmd="module purge; module load openmpi/4.0.4 gpu/0.15.4
                                 amber/20;
                                 conda activate westpa-2.0")

2. get_server_mode()
   → confirms tleap ✓, squeue ✓, pmemd.cuda ✓ on EXPANSE

3. fetch_pdb_structure(pdb_id="1UAO", output_dir="./chignolin")
   → downloads 1UAO.pdb locally

4. prepare_amber_system(pdb_file="./chignolin/1UAO.pdb",
                        output_dir="/expanse/lustre/scratch/ssonti/chignolin/amber_prep")
   → uploads PDB via SFTP, runs tleap on EXPANSE,
     returns remote prmtop/rst7 paths

5. validate_simulation_config(method="parmetad", backend="amber",
                              nsteps=50000, ntpr=500, ...)
   → pcoord_len = 50000/500 + 1 = 101 ✓

6. generate_simulation_files(...,
      output_dir="/expanse/lustre/scratch/ssonti/chignolin/we_sim")
   → writes west.cfg, runseg.sh, run_WE.sh, plumed.dat, md.in etc.
     directly to EXPANSE via SFTP

7. disconnect_from_cluster()
```

### Step 3 — Launch on EXPANSE

```bash
cd /expanse/lustre/scratch/ssonti/chignolin/we_sim
bash init.sh
sbatch run_WE.sh
```

---

## Parameters Collected During Setup

| Parameter | Example | Notes |
|-----------|---------|-------|
| `method` | `parmetad` | parmetad or parGaMD |
| `backend` | `amber` | amber or openmm |
| `scratch_dir` | `/expanse/lustre/scratch/user/proj` | Lustre scratch only |
| `conda_env` | `westpa2` | Conda env with WESTPA |
| `account` | `ucd192` | SLURM billing account |
| `partition` | `gpu-shared` | EXPANSE: gpu-shared or gpu |
| `nodes` | `4` | Compute nodes |
| `gpus_per_node` | `4` | GPUs per node = walkers per node |
| `walltime` | `24:00:00` | SLURM walltime |
| `amberhome` | `/home/user/amber22` | AMBER installation |
| `plumed_kernel_dir` | `/home/user/plumed2/lib` | ParMetaD: libplumedKernel.so dir |
| `temperature` | `300` | Kelvin |
| `nsteps` | `50000` | MD steps per WE segment (must divide by ntpr) |
| `ntpr` | `500` | Output frequency |
| `bin_boundaries` | `[0,0.5,1,2,3,4,5]` | Progress coordinate bins |
| `bin_target_counts` | `4` | Walkers per bin |
| `max_iterations` | `200` | WE iterations |

### Critical constraint

```
pcoord_len = nsteps / ntpr + 1
```

`nsteps` must be **exactly divisible** by `ntpr`. The server validates this before generating any files.

---

## Generated Files

```
west.cfg               # WESTPA configuration (pcoord_len, bins, iterations)
runseg.sh              # Segment propagation (calls AMBER + PLUMED)
run_WE.sh              # Main SLURM submission script (ZMQ master)
node.sh                # Worker node startup (ZMQ workers)
env.sh                 # Module loads + conda activate
md.in                  # AMBER MD input (nstlim, dt, ntpr, etc.)
plumed.dat             # PLUMED metadynamics (ParMetaD only)
cMD/run_cmd.sh         # cGaMD pre-run (ParGaMD only)
SETUP_INSTRUCTIONS.md  # Step-by-step sbatch guide
config_used.json       # Full parameter record
```

---

## Execution Modes

The server adapts automatically — no code changes needed:

| Mode | How triggered | What runs where |
|------|--------------|-----------------|
| **SSH-cluster** | `connect_to_cluster(..., use_control_master=true)` | tleap + file writes on EXPANSE via SSH/SFTP |
| **Local-cluster** | tleap in local PATH (no SSH) | tleap runs locally |
| **Laptop** | No SSH, no local tleap | Generates templates + tleap input for manual use |

---

## Troubleshooting

**`Failed to connect` on MCP list:**
- Check the Python path is correct (full absolute path)
- Verify `pip install -r requirements.txt` completed without errors
- Test the server manually: `python mcp_server.py` should hang silently (awaiting stdio)

**SSH ControlMaster socket not found:**
- Make sure `ssh expanse` is running in another terminal
- Verify `~/.ssh/config` has the `ControlPath` line
- Check socket exists: `ls ~/.ssh/cm-*`

**TOTP still being requested:**
- The ControlMaster socket must already exist before calling `connect_to_cluster`
- If it expired (>8 hours), open a new `ssh expanse` terminal

**`tleap not found` on EXPANSE:**
- Add `module load amber/22` to `setup_cmd` in `connect_to_cluster`
- Or load it in the ControlMaster terminal before connecting:  
  `ssh expanse "module load amber && which tleap"`

**Files not appearing on EXPANSE:**
- Confirm `output_dir` is an absolute path on the cluster
- Check SFTP write permissions: `ssh expanse "ls -la /expanse/lustre/scratch/ssonti/"`

---

## EXPANSE-Specific Reference

```bash
# Recommended setup_cmd for connect_to_cluster:
"module purge; module load gpu/0.15.4 amber/22 anaconda3/2021.05; conda activate westpa2"

# Scratch filesystem (required — home dir NFS is too slow):
/expanse/lustre/scratch/$USER/your_project

# Submit jobs:
sbatch run_WE.sh

# ParGaMD: run cGaMD pre-run first:
sbatch cMD/run_cmd.sh
# Then with dependency:
sbatch --dependency=afterok:JOBID run_WE.sh

# Monitor:
squeue -u $USER
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS

# Check simulation progress:
python3 -c "import h5py; f=h5py.File('west.h5'); print('Iterations:', len(f['iterations']))"
```

---

## How CLAUDE.md Works

`CLAUDE.md` is a domain knowledge file that any Claude session loads automatically as context. It contains:
- Full parameter reference
- pcoord_len constraint explanation and examples
- ParMetaD PLUMED.dat key fields
- ParGaMD two-stage workflow
- EXPANSE module setup and job submission
- Debugging tables for common WESTPA and AMBER errors
- FAQs

This replaces a RAG system — Claude reads it once per session and has full expert knowledge without a vector database or GPU.
