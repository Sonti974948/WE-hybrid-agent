"""
tools/file_generator.py
Wraps generators.py to produce ParMetaD / ParGaMD simulation files.

Uses the global Executor singleton so generated files are written either:
  • Locally  — when no SSH session is active (pathlib writes)
  • Remotely — over SFTP when connected to a cluster (paramiko SFTP)

Works in both cluster and laptop modes — no executables required for
file generation itself.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

from tools.executor import get_executor

# ── Import generators from parent directory ───────────────────────────────────
_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))

try:
    from generators import generate_files, compute_pcoord_len
    _GEN_AVAILABLE = True
    _GEN_ERROR     = ""
except ImportError as e:
    _GEN_AVAILABLE = False
    _GEN_ERROR     = str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _amber_version_from_path(amberhome: str) -> str:
    """Extract AMBER major version number from an AMBERHOME path, e.g. '20' from '.../amber-20-...'"""
    m = re.search(r'amber[/-](\d+)', amberhome, re.IGNORECASE)
    return m.group(1) if m else "22"


# ── Config builder ────────────────────────────────────────────────────────────

def _build_config_dict(params: dict) -> dict:
    """
    Convert flat MCP tool arguments into the nested dict generators.py expects.
    Only includes keys that were provided (non-None).
    """
    p = params

    def _get(*keys, default=None):
        for k in keys:
            if k in p and p[k] is not None:
                return p[k]
        return default

    method  = _get("method",  default="parmetad").lower()
    backend = _get("backend", default="amber").lower()

    cfg = {
        "method":  method,
        "backend": backend,

        "hpc": {
            "scratch_dir":   _get("scratch_dir"),
            "conda_env":     _get("conda_env"),
            "account":       _get("account"),
            "partition":     _get("partition"),
            "nodes":         int(_get("nodes",         default=4)),
            "gpus_per_node": int(_get("gpus_per_node", default=4)),
            "walltime":      _get("walltime",    default="24:00:00"),
            "email":         _get("email",       default=""),
            "mem":           _get("mem",         default=""),
            "cuda_version":  _get("cuda_version", default="11.7"),
            "amber_version": _get("amber_version", default=_amber_version_from_path(_get("amberhome", default=""))),
        },

        "paths": {
            "amberhome":         _get("amberhome",         default="/path/to/amber"),
            "plumed_kernel_dir": _get("plumed_kernel_dir", default="/path/to/plumed/lib"),
        },

        "md": {
            "temperature": float(_get("temperature", default=300.0)),
            "timestep":    float(_get("timestep",    default=0.002)),
            "nsteps":      int(_get("nsteps",        default=50000)),
            "ntpr":        int(_get("ntpr",          default=500)),
            "cmd_nsteps":  int(_get("cmd_nsteps",    default=5000000)),
        },

        "westpa": {
            "pcoord_ndim":       int(_get("pcoord_ndim",       default=1)),
            "bin_boundaries":    _get("bin_boundaries",         default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
            "bin_target_counts": int(_get("bin_target_counts",  default=4)),
            "max_iterations":    int(_get("max_iterations",     default=200)),
        },

        "gamd": {
            "sigma0D": float(_get("sigma0D",   default=6.0)),
            "sigma0P": float(_get("sigma0P",   default=6.0)),
            "iE":      int(_get("gamd_iE", "iE", default=2)),
        },
    }

    # Auto-compute pcoord_len
    nsteps = cfg["md"]["nsteps"]
    ntpr   = cfg["md"]["ntpr"]
    cfg["md"]["pcoord_len"] = nsteps // ntpr + 1

    # segs_per_state = total walker count
    cfg["westpa"]["segs_per_state"] = cfg["hpc"]["nodes"] * cfg["hpc"]["gpus_per_node"]

    # bin_boundaries: fmt_bins expects list-of-lists (one per CV dimension)
    raw_bins    = cfg["westpa"]["bin_boundaries"]
    pcoord_ndim = cfg["westpa"]["pcoord_ndim"]
    if raw_bins and not isinstance(raw_bins[0], list):
        bins_with_inf = list(raw_bins) + ['inf']
        cfg["westpa"]["bin_boundaries"] = [bins_with_inf] * pcoord_ndim

    # pcoord / CV section consumed by gen_plumed_dat_metad
    hills_pace = ntpr
    if pcoord_ndim == 1:
        default_cvs = [{"name": "cv1",  "type": "TORSION", "atoms": "1,2,3,4", "sigma": 0.1, "height": 1.2}]
    else:
        default_cvs = [
            {"name": "phi", "type": "TORSION", "atoms": "1,2,3,4", "sigma": 0.1, "height": 1.2},
            {"name": "psi", "type": "TORSION", "atoms": "5,6,7,8", "sigma": 0.1, "height": 1.2},
        ]
    cfg["pcoord"] = {
        "cvs":        _get("cvs", default=default_cvs),
        "hills_pace": hills_pace,
        "biasfactor": int(_get("biasfactor", default=10)),
    }

    return cfg


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_params(params: dict) -> list[str]:
    """Return a list of validation error strings. Empty list = valid."""
    errors  = []
    method  = (params.get("method")  or "").lower()
    backend = (params.get("backend") or "").lower()

    if method not in ("parmetad", "pargamd"):
        errors.append(f"method must be 'parmetad' or 'parGaMD', got: {method!r}")
    if backend not in ("amber", "openmm"):
        errors.append(f"backend must be 'amber' or 'openmm', got: {backend!r}")
    if method == "pargamd" and backend != "amber":
        errors.append("ParGaMD only supports AMBER backend")

    nsteps = params.get("nsteps")
    ntpr   = params.get("ntpr")
    if nsteps and ntpr:
        if int(nsteps) % int(ntpr) != 0:
            nearest = (int(nsteps) // int(ntpr)) * int(ntpr)
            errors.append(
                f"nsteps ({nsteps}) must be divisible by ntpr ({ntpr}). "
                f"Suggested nsteps: {nearest}"
            )

    return errors


# ── Public tool functions ─────────────────────────────────────────────────────

def generate_simulation_files(params: dict, output_dir: str) -> dict:
    """
    Generate all simulation files for ParMetaD or ParGaMD.

    Files are written to output_dir via the Executor:
      • Local path when no SSH session is active
      • Remote path over SFTP when connected to a cluster

    params: flat dict of simulation parameters (see CLAUDE.md)
    output_dir: directory to write files into (local or remote absolute path)

    Returns: {"success": bool, "files": [...], "message": str, "config": {...}}
    """
    if not _GEN_AVAILABLE:
        return {
            "success": False,
            "message": (
                f"generators.py not found: {_GEN_ERROR}\n"
                "Ensure generators.py is in the same directory as mcp_server.py."
            ),
            "files": [],
        }

    # Validate parameters
    errors = _validate_params(params)
    if errors:
        return {"success": False, "message": "\n".join(errors), "files": []}

    # Build nested config
    config = _build_config_dict(params)

    # Generate file contents (pure Python, no I/O)
    try:
        files_dict = generate_files(config)
    except Exception as e:
        return {"success": False, "message": f"Generation error: {e}", "files": []}

    # Write files via Executor (local or SFTP)
    exec_ = get_executor()
    exec_.make_dir(output_dir)

    written = []
    failed  = []
    for rel_path, content in files_dict.items():
        # rel_path may include subdirs like "cMD/run_cmd.sh"
        parts = Path(rel_path).parts
        if len(parts) > 1:
            subdir = output_dir.rstrip("/") + "/" + "/".join(parts[:-1])
            exec_.make_dir(subdir)

        dest = output_dir.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
        try:
            # Make shell scripts executable
            file_mode = 0o755 if rel_path.endswith(".sh") else 0o644
            exec_.write_file(dest, content, mode=file_mode)
            written.append(str(rel_path))
        except Exception as e:
            failed.append(f"{rel_path}: {e}")

    # Write config summary
    config_json = json.dumps(config, indent=2)
    config_dest = output_dir.rstrip("/") + "/config_used.json"
    try:
        exec_.write_file(config_dest, config_json)
        written.append("config_used.json")
    except Exception as e:
        failed.append(f"config_used.json: {e}")

    pcoord_len = config["md"].get("pcoord_len", "?")
    nsteps     = config["md"]["nsteps"]
    ntpr       = config["md"]["ntpr"]

    mode_note = ""
    if exec_.connected:
        mode_note = f" (written to {exec_._host}:{output_dir} via SFTP)"

    message = (
        f"Generated {len(written)} files in: {output_dir}{mode_note}\n"
        f"pcoord_len = {nsteps}/{ntpr} + 1 = {pcoord_len}\n\n"
        f"Next steps:\n"
        f"1. Copy your system files (.prmtop, .pdb, .rst7) to {output_dir}/common_files/\n"
        f"2. Copy basis state restart to {output_dir}/bstates/bstate.rst\n"
        f"3. Read {output_dir}/SETUP_INSTRUCTIONS.md for exact sbatch commands"
    )
    if failed:
        message += "\n\nFailed to write:\n" + "\n".join(f"  • {f}" for f in failed)

    return {
        "success": len(failed) == 0,
        "message":    message,
        "files":      written,
        "failed":     failed,
        "config":     config,
        "output_dir": output_dir,
    }


def validate_config(params: dict) -> dict:
    """
    Validate simulation parameters without generating files.
    Returns missing fields and constraint violations.
    """
    errors  = _validate_params(params)
    missing = []

    required = [
        ("method",            "simulation method (parmetad or parGaMD)"),
        ("backend",           "MD engine (amber or openmm)"),
        ("scratch_dir",       "scratch directory on cluster"),
        ("conda_env",         "conda environment name"),
        ("account",           "SLURM billing account"),
        ("partition",         "SLURM partition"),
        ("nodes",             "number of nodes"),
        ("gpus_per_n