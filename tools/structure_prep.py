"""
tools/structure_prep.py
Fetch PDB structures from RCSB and prepare AMBER topology with tleap.

Uses the global Executor singleton so all cluster operations are
transparently routed through SSH (when connected) or run locally.

Modes
-----
SSH-cluster:  executor connected → PDB fetched locally, tleap runs on EXPANSE,
              prmtop/rst7 written to remote path via SFTP.
Local-cluster: tleap in local PATH → runs directly.
Laptop:        no tleap anywhere → returns tleap input script for manual use.
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

from tools.executor import get_executor


# ─────────────────────────────────────────────────────────────────────────────
# PDB fetch  (always runs locally — RCSB download is fast from a laptop)
# ─────────────────────────────────────────────────────────────────────────────

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str, output_dir: str = ".") -> dict:
    """
    Download a PDB structure from RCSB Protein Data Bank.

    Always runs locally (HTTP download). The PDB file is saved to a local
    path; if SSH is active, the caller or generate flow should upload it.

    Returns: {"success": bool, "pdb_file": str, "message": str, "info": str}
    """
    if not _REQUESTS_OK:
        return {
            "success": False,
            "message": "requests package not installed. Run: pip install requests",
        }

    pdb_id = pdb_id.upper().strip()
    if not re.match(r"^[A-Z0-9]{4}$", pdb_id):
        return {
            "success": False,
            "message": f"Invalid PDB ID: {pdb_id!r}. Must be 4 alphanumeric characters.",
        }

    # Always write locally (laptop side) — we'll upload via SFTP if needed
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{pdb_id}.pdb"

    url = RCSB_URL.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)
    except Exception as e:
        return {"success": False, "message": f"Failed to download {pdb_id} from RCSB: {e}"}

    lines  = resp.text.splitlines()
    header = next((l for l in lines if l.startswith("HEADER")), "")
    title  = " ".join(l[10:].strip() for l in lines if l.startswith("TITLE"))
    atoms  = sum(1 for l in lines if l.startswith("ATOM"))
    chains = sorted(set(l[21] for l in lines if l.startswith("ATOM")))

    info = (
        f"PDB: {pdb_id}\n"
        f"Header: {header[10:].strip()}\n"
        f"Title: {title}\n"
        f"ATOM records: {atoms}\n"
        f"Chains: {', '.join(chains)}"
    )

    return {
        "success": True,
        "pdb_file": str(dest.resolve()),
        "pdb_id": pdb_id,
        "info": info,
        "message": f"Downloaded {pdb_id}.pdb to {dest}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# tleap input generation
# ─────────────────────────────────────────────────────────────────────────────

FF_SOURCES = {
    "ff14SB":       ["leaprc.protein.ff14SB"],
    "ff19SB":       ["leaprc.protein.ff19SB"],
    "ff14SB+GAFF2": ["leaprc.protein.ff14SB", "leaprc.gaff2"],
}

WATER_SOURCES = {
    "TIP3P":    ("leaprc.water.tip3p",   "TIP3PBOX"),
    "TIP4P-Ew": ("leaprc.water.tip4pew", "TIP4PEWBOX"),
    "OPC":      ("leaprc.water.opc",     "OPCBOX"),
    "TIP3P-FB": ("leaprc.water.fb3",     "TIP3PFBOX"),
}


def generate_tleap_input(
    pdb_file: str,
    force_field: str = "ff14SB",
    water_model: str = "TIP3P",
    box_size: float = 10.0,
    box_shape: str = "cubic",
    neutralize: bool = True,
    output_prefix: str = "system",
) -> str:
    """Return a tleap input script as a string."""
    ff_sources          = FF_SOURCES.get(force_field, FF_SOURCES["ff14SB"])
    water_src, water_box = WATER_SOURCES.get(water_model, WATER_SOURCES["TIP3P"])

    lines = []
    for src in ff_sources:
        lines.append(f"source {src}")
    lines.append(f"source {water_src}")
    lines += ["", f"mol = loadpdb {pdb_file}", "", "check mol", ""]

    if box_shape == "octahedral":
        lines.append(f"solvateoct mol {water_box} {box_size:.1f}")
    else:
        lines.append(f"solvatebox mol {water_box} {box_size:.1f}")
    lines.append("")

    if neutralize:
        lines += ["addions mol Na+ 0", "addions mol Cl- 0", ""]

    lines += [
        f"saveamberparm mol {output_prefix}.prmtop {output_prefix}.rst7",
        f"savepdb mol {output_prefix}_solvated.pdb",
        "quit",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# System preparation (tleap)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_amber_system(
    pdb_file: str,
    output_dir: str = ".",
    force_field: str = "ff14SB",
    water_model: str = "TIP3P",
    box_size: float = 10.0,
    box_shape: str = "cubic",
    neutralize: bool = True,
    output_prefix: str = "system",
) -> dict:
    """
    Prepare an AMBER system from a PDB file.

    SSH-cluster mode:  uploads PDB + leap.in to remote output_dir,
                       runs tleap there, reports remote prmtop/rst7 paths.
    Local-cluster mode: runs tleap locally.
    Laptop mode:        returns tleap input for manual execution on cluster.

    Returns a result dict with keys:
      success, mode, tleap_input, tleap_input_file,
      prmtop, rst7, tleap_log, message, next_steps
    """
    exec_ = get_executor()

    # ── Resolve PDB path ──────────────────────────────────────────────────────
    pdb_local = Path(pdb_file).resolve()
    if not pdb_local.exists():
        return {"success": False, "message": f"PDB file not found: {pdb_file}"}

    # ── Determine where tleap will run and what output_dir means ─────────────
    if exec_.connected:
        mode = "ssh"
        remote_dir = output_dir  # remote absolute path (caller must provide)
        tleap_bin  = exec_.which("tleap")
    else:
        mode = "local"
        tleap_bin  = shutil.which("tleap")
        remote_dir = None

    has_tleap = tleap_bin is not None

    # ── Generate tleap input ─────────────────────────────────────────────────
    # For remote runs, PDB will be at remote_dir/<pdb_name>.pdb after upload.
    if mode == "ssh":
        remote_pdb = f"{remote_dir}/{pdb_local.name}"
        leap_prefix = f"{remote_dir}/{output_prefix}"
    else:
        local_out = Path(output_dir).resolve()
        leap_prefix = str(local_out / output_prefix)
        remote_pdb  = str(pdb_local)

    leap_input = generate_tleap_input(
        pdb_file=remote_pdb if mode == "ssh" else str(pdb_local),
        force_field=force_field,
        water_model=water_model,
        box_size=box_size,
        box_shape=box_shape,
        neutralize=neutralize,
        output_prefix=leap_prefix,
    )

    # ── Laptop mode: no tleap, return template ────────────────────────────────
    if not has_tleap:
        # Save leap.in locally so user can inspect/copy it
        local_out = Path(output_dir)
        local_out.mkdir(parents=True, exist_ok=True)
        leap_in_local = local_out / "leap.in"
        leap_in_local.write_text(leap_input)

        if mode == "ssh":
            # SSH connected but tleap not found on remote — upload file anyway
            exec_.make_dir(remote_dir)
            exec_.write_file(f"{remote_dir}/leap.in", leap_input)
            exec_.write_file(remote_pdb, pdb_local.read_text())
            next_steps = (
                f"tleap not found on {exec_._host}. Files uploaded to {remote_dir}/.\n\n"
                f"On the login node, load AMBER and run:\n"
                f"  module load amber\n"
                f"  cd {remote_dir}\n"
                f"  tleap -f leap.in\n\n"
                f"This will produce:\n"
                f"  {output_prefix}.prmtop\n"
                f"  {output_prefix}.rst7\n"
                f"  {output_prefix}_solvated.pdb\n"
            )
        else:
            next_steps = (
                f"tleap is not available locally. Copy leap.in to EXPANSE and run:\n\n"
                f"  module load amber\n"
                f"  tleap -f leap.in\n\n"
                f"Output files will be:\n"
                f"  {output_prefix}.prmtop\n"
                f"  {output_prefix}.rst7\n"
            )

        return {
            "success": True,
            "mode":    "laptop" if mode == "local" else "ssh-no-tleap",
            "tleap_input":      leap_input,
            "tleap_input_file": str(leap_in_local),
            "prmtop":    None,
            "rst7":      None,
            "tleap_log": None,
            "message":   f"Generated tleap input at {leap_in_local}",
            "next_steps": next_steps,
        }

    # ── Full execution mode ───────────────────────────────────────────────────
    if mode == "ssh":
        # Upload PDB and leap.in to cluster
        exec_.make_dir(remote_dir)
        exec_.write_file(remote_pdb, pdb_local.read_text())
        exec_.write_file(f"{remote_dir}/leap.in", leap_input)

        rc, stdout, stderr = exec_.run_command(
            f"tleap -f leap.in",
            cwd=remote_dir,
            timeout=120,
        )
        log = stdout + stderr

        # Check for prmtop on remote
        prmtop_remote = f"{remote_dir}/{output_prefix}.prmtop"
        rst7_remote   = f"{remote_dir}/{output_prefix}.rst7"

        if rc != 0 or not exec_.file_exists(prmtop_remote):
            error_lines = [l for l in log.splitlines()
                           if any(w in l.upper() for w in ("ERROR","FATAL","FAILED","UNRECOGNIZED"))]
            return {
                "success": False,
                "mode":    "ssh",
                "tleap_input":      leap_input,
                "tleap_input_file": f"{remote_dir}/leap.in",
                "tleap_log": log,
                "message": (
                    f"tleap failed on {exec_._host}.\n"
                    + "\n".join(error_lines[:10])
                ),
            }

        next_steps = (
            f"System prepared on {exec_._host}.\n"
            f"  Topology:    {prmtop_remote}\n"
            f"  Coordinates: {rst7_remote}\n\n"
            f"Copy these to your simulation directory on the cluster:\n"
            f"  cp {prmtop_remote} <sim_dir>/common_files/\n"
            f"  cp {rst7_remote}   <sim_dir>/bstates/bstate.rst\n"
            f"  cp {rst7_remote}   <sim_dir>/common_files/\n"
        )
        return {
            "success": True,
            "mode":    "ssh",
            "tleap_input":      leap_input,
            "tleap_input_file": f"{remote_dir}/leap.in",
            "prmtop":  prmtop_remote,
            "rst7":    rst7_remote,
            "tleap_log": log,
            "message": f"tleap completed on {exec_._host}.",
            "next_steps": next_steps,
        }

    else:
        # Local cluster mode
        local_out = Path(output_dir).resolve()
        local_out.mkdir(parents=True, exist_ok=True)
        leap_in_path = local_out / "leap.in"
        leap_in_path.write_text(leap_input)
        log_path = local_out / "leap.log"

        rc, stdout, stderr = exec_.run_command(
            f"tleap -f {leap_in_path}",
            cwd=str(local_out),
            timeout=120,
        )
        log = stdout + stderr
        log_path.write_text(log)

        prmtop = local_out / f"{output_prefix}.prmtop"
        rst7   = local_out / f"{output_prefix}.rst7"

        if not prmtop.exists():
            error_lines = [l for l in log.splitlines()
                           if any(w in l.upper() for w in ("ERROR","FATAL","FAILED","UNRECOGNIZED"))]
            return {
                "success": False,
                "mode":    "local",
                "tleap_input":      leap_input,
                "tleap_input_file": str(leap_in_path),
                "tleap_log": log,
                "message": (
                    f"tleap did not produce {output_prefix}.prmtop.\n"
                    + "\n".join(error_lines[:10])
                ),
            }

        next_steps = (
            f"System prepared locally.\n"
            f"  Topology:    {prmtop}\n"
            f"  Coordinates: {rst7}\n\n"
            f"Copy these to your simulation directory:\n"
            f"  cp {prmtop} <sim_dir>/common_files/\n"
            f"  cp {rst7}   <sim_dir>/bstates/bstate.rst\n"
            f"  cp {rst7}   <sim_dir>/common_files/\n"
        )
        return {
            "success": True,
            "mode":    "local",
            "tleap_input":      leap_input,
            "tleap_input_file": str(leap_in_path),
            "prmtop":  str(prmtop),
            "rst7":    str(rst7),
            "tleap_log": log,
            "message": f"tleap completed. prmtop: {prmtop.stat().st_size // 1024} KB",
            "next_steps": next_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode info (used by get_server_mode MCP tool)
# ─────────────────────────────────────────────────────────────────────────────

def get_mode_info() -> dict:
    """Return current execution mode and available capabilities."""
    exec_ = get_executor()

    if exec_.connected:
        caps = exec_.get_capabilities()
        mode = "ssh-cluster" if caps.get("squeue") else "ssh-local"
        notes = []
        if not caps.get("tleap"):
            notes.append("tleap not found on remote — structure prep will return templates")
        if not caps.get("cpptraj"):
            notes.append("cpptraj not found on remote — CV generation will return templates")
        if not caps.get("squeue"):
            notes.append("squeue not found on remote — may not be a SLURM cluster")
        conn = exec_.connection_info()
        return {
            "mode":          mode,
            "connected":     True,
            "host":          conn["host"],
            "username":      conn["username"],
            "capabilities":  caps,
            "notes":         notes,
            "tleap_bin":     exec_.which("tleap"),
            "cpptraj_bin":   exec_.which("cpptraj"),
        }

    # Local / laptop
    tleap_bin   = shutil.which("tleap")
    cpptraj_bin = shutil.which("cpptraj")
    squeue_bin  = shutil.which("squeue")
    caps = {
        "tleap":      tleap_bin   is not None,
        "cpptraj":    cpptraj_bin is not None,
        "squeue":     squeue_bin  is not None,
        "pmemd.cuda": shutil.which("pmemd.cuda") is not None,
    }
    mode = "cluster" if caps["tleap"] and caps["squeue"] else (
           "local"   if caps["tleap"] else "laptop")

    notes = []
    if not caps["tleap"]:
        notes.append("tleap not found — use connect_to_cluster or run on EXPANSE directly")
    if not caps["cpptraj"]:
        notes.append("cpptraj not found — CV generation will return templates only")
    if not caps["squeue"]:
        notes.append("squeue not found — not running on a SLURM cluster")

    return {
        "mode":        mode,
        "connected":   False,
        "host":        None,
        "username":    None,
        "capabilities": caps,
        "notes":       notes,
        "tleap_bin":   tleap_bin,
        "cpptraj_bin": cpptraj_bin,
    }
