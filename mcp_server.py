"""
mcp_server.py — WE-Hybrid MCP Server
Helps researchers set up ParMetaD and ParGaMD simulations via Claude.

Auto-detects environment:
  SSH mode   (connected): all commands run on remote cluster via paramiko
  Cluster mode (local):   tleap + squeue in PATH → full local execution
  Laptop mode:            no cluster tools → generates templates

Install:
  pip install mcp requests pydantic paramiko

Add to Claude Code config (~/.claude/claude_code_config.json or similar):
  {
    "mcpServers": {
      "we-hybrid": {
        "command": "python",
        "args": ["/path/to/we_hybrid_mcp/mcp_server.py"]
      }
    }
  }

Or via CLI:
  claude mcp add we-hybrid python /path/to/we_hybrid_mcp/mcp_server.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# ── MCP SDK ───────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# ── Local tools ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from tools.file_generator import generate_simulation_files, validate_config
from tools.structure_prep import fetch_pdb, prepare_amber_system, get_mode_info
from tools.executor import get_executor

# ── Server ────────────────────────────────────────────────────────────────────
app = Server("we-hybrid-mcp")


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        # ── 0a. SSH connect ───────────────────────────────────────────────────
        types.Tool(
            name="connect_to_cluster",
            description=(
                "Open an SSH connection from this laptop to an HPC cluster. "
                "Two modes:\n"
                "  use_control_master=True (RECOMMENDED for EXPANSE/DUO clusters): "
                "piggybacks on an existing 'ssh expanse' terminal session — "
                "no TOTP per command. Requires ~/.ssh/config ControlMaster setup "
                "and an active ssh session in another terminal.\n"
                "  use_control_master=False (default): uses paramiko directly — "
                "works for clusters without MFA.\n"
                "Call get_server_mode after connecting to verify capabilities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": (
                            "SSH host alias (from ~/.ssh/config) or full hostname. "
                            "For EXPANSE with ControlMaster: use the alias 'expanse'. "
                            "For direct paramiko: use 'login.expanse.sdsc.edu'."
                        ),
                    },
                    "username": {
                        "type": "string",
                        "description": "Your cluster username, e.g. ssonti",
                    },
                    "use_control_master": {
                        "type": "boolean",
                        "description": (
                            "If true, reuse an existing SSH ControlMaster socket "
                            "(recommended for EXPANSE — avoids TOTP per command). "
                            "Requires an active 'ssh expanse' session in another terminal. "
                            "Default: false."
                        ),
                    },
                    "control_path": {
                        "type": "string",
                        "description": (
                            "Path to the ControlMaster socket (auto-detected if omitted). "
                            "Only used when use_control_master=true."
                        ),
                    },
                    "key_file": {
                        "type": "string",
                        "description": "Path to SSH private key. Only used when use_control_master=false.",
                    },
                    "password": {
                        "type": "string",
                        "description": "SSH password. Only used when use_control_master=false.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "SSH port (default: 22). Only used when use_control_master=false.",
                    },
                    "setup_cmd": {
                        "type": "string",
                        "description": (
                            "Shell commands prepended to every remote command. "
                            "For EXPANSE: 'module purge; module load gpu/0.15.4 "
                            "amber/22 anaconda3/2021.05; conda activate westpa2'"
                        ),
                    },
                },
                "required": ["host"],
            },
        ),

        # ── 0b. SSH disconnect ────────────────────────────────────────────────
        types.Tool(
            name="disconnect_from_cluster",
            description=(
                "Close the active SSH connection to the cluster. After this, "
                "all tool calls will run locally again."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        # ── 0c. Mode check ────────────────────────────────────────────────────
        types.Tool(
            name="get_server_mode",
            description=(
                "Returns the current execution mode and available capabilities: "
                "SSH-connected cluster, local cluster, or laptop (template-only). "
                "Call this first — or after connect_to_cluster — to understand "
                "what the server can do in the current environment."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        # ── 1. File generation ────────────────────────────────────────────────
        types.Tool(
            name="validate_simulation_config",
            description=(
                "Validate simulation parameters before generating files. "
                "Returns missing required fields, constraint violations (e.g. "
                "nsteps not divisible by ntpr), and the computed pcoord_len. "
                "Call this before generate_simulation_files."
            ),
            inputSchema={
                "type": "object",
                "properties": _sim_param_schema(),
                "required": [],
            },
        ),

        types.Tool(
            name="generate_simulation_files",
            description=(
                "Generate all files needed for a ParMetaD or ParGaMD WE simulation: "
                "west.cfg, runseg.sh, run_WE.sh, node.sh, env.sh, plumed.dat (ParMetaD), "
                "md.in (AMBER), cMD/run_cmd.sh (ParGaMD), and SETUP_INSTRUCTIONS.md. "
                "If SSH is connected, files are written directly to the cluster via SFTP. "
                "Otherwise, files are created locally. "
                "Call validate_simulation_config first to catch errors early."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    **_sim_param_schema(),
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Directory to write files into. "
                            "If SSH-connected, use an absolute path on the cluster "
                            "e.g. /expanse/lustre/scratch/ssonti/chignolin/we_sim. "
                            "If local, use a relative or absolute local path."
                        ),
                    },
                },
                "required": ["method", "backend", "output_dir"],
            },
        ),

        # ── 2. Structure preparation ──────────────────────────────────────────
        types.Tool(
            name="fetch_pdb_structure",
            description=(
                "Download a protein structure from the RCSB Protein Data Bank by PDB ID. "
                "Example: '1UAO' for Chignolin, '1L2Y' for Trp-cage. "
                "The file is saved locally. If SSH is connected, you can then call "
                "prepare_amber_system with a remote output_dir to upload and run tleap."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-character PDB code, e.g. '1UAO'",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Local directory to save the PDB file (default: current dir)",
                    },
                },
                "required": ["pdb_id"],
            },
        ),

        types.Tool(
            name="prepare_amber_system",
            description=(
                "Prepare an AMBER topology (prmtop + rst7) from a PDB file using tleap. "
                "SSH-cluster mode: uploads PDB to the cluster, runs tleap remotely, "
                "  returns remote prmtop/rst7 paths — no local AMBER needed. "
                "Local-cluster mode: runs tleap locally. "
                "Laptop mode: returns tleap input script to run manually on the cluster. "
                "Supports ff14SB, ff19SB force fields and TIP3P, OPC water models."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pdb_file": {
                        "type": "string",
                        "description": "Local path to the PDB file (e.g. from fetch_pdb_structure)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Directory to write tleap files and output. "
                            "If SSH-connected, use an absolute path on the cluster "
                            "e.g. /expanse/lustre/scratch/ssonti/chignolin/amber_prep. "
                            "The PDB file will be uploaded automatically."
                        ),
                    },
                    "force_field": {
                        "type": "string",
                        "enum": ["ff14SB", "ff19SB", "ff14SB+GAFF2"],
                        "description": "AMBER force field (default: ff14SB)",
                    },
                    "water_model": {
                        "type": "string",
                        "enum": ["TIP3P", "TIP4P-Ew", "OPC", "TIP3P-FB"],
                        "description": "Water model for solvation (default: TIP3P)",
                    },
                    "box_size": {
                        "type": "number",
                        "description": "Solvent box padding in Angstroms (default: 10.0)",
                    },
                    "box_shape": {
                        "type": "string",
                        "enum": ["cubic", "octahedral"],
                        "description": "Box shape: cubic or octahedral (default: cubic)",
                    },
                    "neutralize": {
                        "type": "boolean",
                        "description": "Add Na+/Cl- ions to neutralize (default: true)",
                    },
                    "output_prefix": {
                        "type": "string",
                        "description": "Prefix for output files, e.g. 'chignolin' → chignolin.prmtop (default: system)",
                    },
                },
                "required": ["pdb_file"],
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:

    def _text(data) -> list[types.TextContent]:
        if isinstance(data, str):
            return [types.TextContent(type="text", text=data)]
        return [types.TextContent(type="text", text=json.dumps(data, indent=2))]

    match name:

        # ── SSH connect ───────────────────────────────────────────────────────
        case "connect_to_cluster":
            exec_ = get_executor()
            use_cm = bool(arguments.get("use_control_master", False))

            if use_cm:
                ok, msg = exec_.connect_control_master(
                    host         = arguments.get("host", ""),
                    username     = arguments.get("username"),
                    control_path = arguments.get("control_path"),
                    setup_cmd    = arguments.get("setup_cmd", ""),
                )
            else:
                ok, msg = exec_.connect(
                    host      = arguments.get("host", ""),
                    username  = arguments.get("username", ""),
                    key_file  = arguments.get("key_file"),
                    password  = arguments.get("password"),
                    port      = int(arguments.get("port", 22)),
                    setup_cmd = arguments.get("setup_cmd", ""),
                )

            if ok:
                info = get_mode_info()
                caps = info.get("capabilities", {})
                cap_lines = [
                    f"  {'✓' if avail else '✗'} {tool}"
                    for tool, avail in caps.items()
                ]
                return _text(
                    f"✅ {msg}\n\n"
                    f"Remote capabilities:\n" + "\n".join(cap_lines) +
                    "\n\nAll tool calls will now run on the cluster.\n"
                    "Call get_server_mode for full details."
                )
            return _text(f"❌ Connection failed:\n\n{msg}")

        # ── SSH disconnect ────────────────────────────────────────────────────
        case "disconnect_from_cluster":
            exec_ = get_executor()
            if not exec_.connected:
                return _text("Not connected to any cluster.")
            host = exec_._host
            exec_.disconnect()
            return _text(f"✅ Disconnected from {host}. Running locally now.")

        # ── Mode check ────────────────────────────────────────────────────────
        case "get_server_mode":
            exec_ = get_executor()
            info  = get_mode_info()
            lines = []

            if exec_.connected:
                lines += [
                    f"Mode: SSH-CLUSTER",
                    f"Host:     {info['host']}",
                    f"User:     {info['username']}",
                    f"",
                    "Remote capabilities:",
                ]
                for tool, avail in info.get("capabilities", {}).items():
                    path = exec_.which(tool)
                    lines.append(f"  {'✓' if avail else '✗'} {tool}"
                                 + (f"  ({path})" if path else ""))
            else:
                lines += [
                    f"Mode: {info['mode'].upper()}",
                    f"tleap:   {'✓ ' + (info['tleap_bin'] or '') if info.get('tleap_bin') else '✗ not found'}",
                    f"cpptraj: {'✓ ' + (info['cpptraj_bin'] or '') if info.get('cpptraj_bin') else '✗ not found'}",
                    f"",
                    "Tip: call connect_to_cluster to use EXPANSE remotely.",
                ]

            if info.get("notes"):
                lines += ["", "Notes:"] + [f"  • {n}" for n in info["notes"]]

            return _text("\n".join(lines))

        # ── Validate config ───────────────────────────────────────────────────
        case "validate_simulation_config":
            result = validate_config(arguments)
            lines  = [result["message"]]
            if result["errors"]:
                lines += ["", "Errors:"] + [f"  ✗ {e}" for e in result["errors"]]
            if result["missing"]:
                lines += ["", "Missing fields:"] + [f"  • {m}" for m in result["missing"]]
            if result["pcoord_len_info"]:
                lines += ["", result["pcoord_len_info"]]
            return _text("\n".join(lines))

        # ── Generate files ────────────────────────────────────────────────────
        case "generate_simulation_files":
            output_dir = arguments.pop("output_dir", "./we_simulation")
            result     = generate_simulation_files(arguments, output_dir)
            if result["success"]:
                lines = [
                    result["message"], "",
                    f"Files written ({len(result['files'])}):",
                ] + [f"  {f}" for f in sorted(result["files"])]
            else:
                lines = [f"❌ {result['message']}"]
            return _text("\n".join(lines))

        # ── Fetch PDB ─────────────────────────────────────────────────────────
        case "fetch_pdb_structure":
            pdb_id     = arguments.get("pdb_id", "")
            output_dir = arguments.get("output_dir", ".")
            result     = fetch_pdb(pdb_id, output_dir)
            if result["success"]:
                exec_ = get_executor()
                next_hint = (
                    f"call prepare_amber_system with pdb_file={result['pdb_file']!r} "
                    f"and output_dir=<remote_path>" if exec_.connected else
                    f"call prepare_amber_system with pdb_file={result['pdb_file']!r}"
                )
                return _text(
                    f"✓ Downloaded: {result['pdb_file']}\n\n"
                    f"{result['info']}\n\n"
                    f"Next step: {next_hint}"
                )
            return _text(f"❌ {result['message']}")

        # ── Prepare AMBER system ──────────────────────────────────────────────
        case "prepare_amber_system":
            result = prepare_amber_system(
                pdb_file      = arguments.get("pdb_file", ""),
                output_dir    = arguments.get("output_dir", "."),
                force_field   = arguments.get("force_field", "ff14SB"),
                water_model   = arguments.get("water_model", "TIP3P"),
                box_size      = float(arguments.get("box_size", 10.0)),
                box_shape     = arguments.get("box_shape", "cubic"),
                neutralize    = bool(arguments.get("neutralize", True)),
                output_prefix = arguments.get("output_prefix", "system"),
            )
            if result["success"]:
                lines = [f"✓ {result['message']}", "", result["next_steps"]]
                mode = result.get("mode", "")
                if "laptop" in mode or "no-tleap" in mode:
                    lines += ["", "tleap input (leap.in):", "```", result["tleap_input"], "```"]
            else:
                lines = [f"❌ {result['message']}"]
                if result.get("tleap_log"):
                    lines += ["", "tleap log (last 20 lines):"]
                    lines += result["tleap_log"].splitlines()[-20:]
            return _text("\n".join(lines))

        case _:
            return _text(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameter schema (shared by validate + generate tools)
# ─────────────────────────────────────────────────────────────────────────────

def _sim_param_schema() -> dict:
    return {
        "method": {
            "type": "string", "enum": ["parmetad", "parGaMD"],
            "description": "Simulation method: parmetad (WESTPA+PLUMED) or parGaMD (WESTPA+GaMD)",
        },
        "backend": {
            "type": "string", "enum": ["amber", "openmm"],
            "description": "MD engine: amber (pmemd.cuda) or openmm. ParGaMD is amber-only.",
        },
        "scratch_dir": {
            "type": "string",
            "description": "Absolute scratch path on cluster, e.g. /expanse/lustre/scratch/user/project",
        },
        "conda_env": {
            "type": "string",
            "description": "Conda environment containing WESTPA",
        },
        "account": {
            "type": "string",
            "description": "SLURM billing account, e.g. ucd192",
        },
        "partition": {
            "type": "string",
            "description": "SLURM partition, e.g. gpu-shared",
        },
        "nodes": {
            "type": "integer",
            "description": "Number of compute nodes",
        },
        "gpus_per_node": {
            "type": "integer",
            "description": "GPUs per node (= walkers per node)",
        },
        "walltime": {
            "type": "string",
            "description": "SLURM walltime, e.g. 24:00:00",
        },
        "amberhome": {
            "type": "string",
            "description": "Path to AMBER installation, e.g. /home/user/amber22",
        },
        "plumed_kernel_dir": {
            "type": "string",
            "description": "Directory containing libplumedKernel.so (ParMetaD only)",
        },
        "temperature": {
            "type": "number",
            "description": "Simulation temperature in Kelvin, e.g. 300",
        },
        "timestep": {
            "type": "number",
            "description": "MD timestep in ps, typically 0.002 (2 fs)",
        },
        "nsteps": {
            "type": "integer",
            "description": "MD steps per WE segment (nstlim). Must be divisible by ntpr.",
        },
        "ntpr": {
            "type": "integer",
            "description": "Output frequency in steps. pcoord_len = nsteps/ntpr + 1.",
        },
        "pcoord_ndim": {
            "type": "integer", "enum": [1, 2],
            "description": "Progress coordinate dimensions: 1 (single CV) or 2 (two CVs)",
        },
        "bin_boundaries": {
            "type": "array", "items": {"type": "number"},
            "description": "List of bin boundary values along pcoord, e.g. [0,1,2,3,4,5]",
        },
        "bin_target_counts": {
            "type": "integer",
            "description": "Target walkers per bin, typically 4-8",
        },
        "max_iterations": {
            "type": "integer",
            "description": "Maximum number of WE iterations",
        },
        "sigma0D": {
            "type": "number",
            "description": "GaMD dihedral boost threshold kcal/mol (ParGaMD only, default 6.0)",
        },
        "sigma0P": {
            "type": "number",
            "description": "GaMD total potential boost threshold kcal/mol (ParGaMD only, default 6.0)",
        },
        "gamd_iE": {
            "type": "integer", "enum": [1, 2],
            "description": "GaMD boost mode: 1=lower bound, 2=upper bound (recommended, default 2)",
        },
        "email": {
            "type": "string",
            "description": "Email for SLURM job notifications (optional)",
        },
        "cmd_nsteps": {
            "type": "integer",
            "description": "Steps for cGaMD pre-run (ParGaMD only, default 5000000)",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
