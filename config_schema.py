"""
config_schema.py
Pydantic models for the WE-Hybrid simulation configuration.
Used for structured extraction from LLM responses and validation.
"""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, field_validator, model_validator


class HPCConfig(BaseModel):
    scratch_dir: str = ""
    conda_env: str = ""
    account: str = ""
    partition: str = ""
    nodes: int = 4
    gpus_per_node: int = 4
    walltime: str = "24:00:00"
    email: str = ""
    cuda_version: str = "11.0.2"
    amber_version: str = "20-patch15"
    mem: str = "50G"
    cmd_partition: str = "gpu-shared"
    cmd_walltime: str = "48:00:00"


class PathsConfig(BaseModel):
    amberhome: str = ""
    plumed_kernel_dir: str = ""


class MDConfig(BaseModel):
    temperature: float = 300.0
    timestep: float = 0.002
    nsteps: int = 50000
    ntpr: int = 500
    pcoord_len: int = 0
    cmd_nsteps: int = 2000000

    @model_validator(mode="after")
    def compute_pcoord_len(self):
        if self.nsteps > 0 and self.ntpr > 0:
            self.pcoord_len = self.nsteps // self.ntpr + 1
        return self

    @field_validator("ntpr")
    @classmethod
    def ntpr_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("ntpr must be positive")
        return v

    @field_validator("timestep")
    @classmethod
    def timestep_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("timestep must be positive")
        return v

    def validate_divisibility(self) -> Optional[str]:
        """Returns error message if nstlim % ntpr != 0, else None."""
        if self.nsteps % self.ntpr != 0:
            return (
                f"nstlim ({self.nsteps}) must be divisible by ntpr ({self.ntpr}). "
                f"Suggested ntpr: {self.nsteps // self.ntpr} or nstlim: {(self.nsteps // self.ntpr) * self.ntpr}"
            )
        return None


class WESTPAConfig(BaseModel):
    pcoord_ndim: int = 1
    bin_boundaries: List[List[Union[float, int, str]]] = []
    bin_target_counts: int = 4
    max_iterations: int = 200
    segs_per_state: int = 4

    @field_validator("pcoord_ndim")
    @classmethod
    def ndim_must_be_1_or_2(cls, v):
        if v not in (1, 2):
            raise ValueError("pcoord_ndim must be 1 or 2")
        return v


class GaMDConfig(BaseModel):
    sigma0D: float = 6.0
    sigma0P: float = 6.0
    iE: int = 2

    @field_validator("iE")
    @classmethod
    def ie_must_be_1_or_2(cls, v):
        if v not in (1, 2):
            raise ValueError("iE must be 1 or 2")
        return v


class CVConfig(BaseModel):
    name: str = "cv1"
    type: str = "RMSD"     # RMSD, DISTANCE, TORSION, GYRATION
    atoms: str = ""
    sigma: float = 0.05
    height: float = 1.2


class PCoordConfig(BaseModel):
    cvs: List[CVConfig] = []
    hills_pace: int = 500
    biasfactor: int = 10


class SimConfig(BaseModel):
    """Master simulation configuration."""
    method: str = ""          # "parmetad" or "parGaMD"
    backend: str = ""         # "amber" or "openmm"
    hpc: HPCConfig = HPCConfig()
    paths: PathsConfig = PathsConfig()
    md: MDConfig = MDConfig()
    westpa: WESTPAConfig = WESTPAConfig()
    gamd: GaMDConfig = GaMDConfig()
    pcoord: PCoordConfig = PCoordConfig()
    tstate: bool = False

    def is_complete(self) -> tuple[bool, list[str]]:
        """
        Check if config has all required fields.
        Returns (complete: bool, missing: list[str])
        """
        missing = []
        if not self.method:
            missing.append("method (parmetad or parGaMD)")
        if not self.backend:
            missing.append("backend (amber or openmm)")
        if not self.hpc.scratch_dir:
            missing.append("HPC scratch directory")
        if not self.hpc.conda_env:
            missing.append("conda environment name")
        if not self.hpc.account:
            missing.append("SLURM account")
        if not self.hpc.partition:
            missing.append("SLURM partition")
        if self.backend == "amber" and not self.paths.amberhome:
            missing.append("AMBERHOME path")
        if not self.westpa.bin_boundaries:
            missing.append("bin boundaries")
        err = self.md.validate_divisibility()
        if err:
            missing.append(f"fix: {err}")
        return len(missing) == 0, missing

    def to_dict(self) -> dict:
        """Convert to the dict format expected by generators.py."""
        return {
            "method": self.method,
            "backend": self.backend,
            "hpc": self.hpc.model_dump(),
            "paths": self.paths.model_dump(),
            "md": self.md.model_dump(),
            "westpa": self.westpa.model_dump(),
            "gamd": self.gamd.model_dump(),
            "pcoord": self.pcoord.model_dump(),
            "tstate": self.tstate,
        }

    def summary_table(self) -> list[tuple[str, str]]:
        """Return a list of (key, value) pairs for display."""
        rows = []
        rows.append(("Method", self.method or "—"))
        rows.append(("Backend", self.backend or "—"))
        rows.append(("Account / Partition", f"{self.hpc.account or '—'} / {self.hpc.partition or '—'}"))
        total_gpus = self.hpc.nodes * self.hpc.gpus_per_node
        rows.append(("Nodes × GPUs", f"{self.hpc.nodes} × {self.hpc.gpus_per_node} = {total_gpus} walkers"))
        rows.append(("Scratch dir", self.hpc.scratch_dir or "—"))
        rows.append(("Conda env", self.hpc.conda_env or "—"))
        rows.append(("Temperature (K)", str(self.md.temperature)))
        rows.append(("nstlim / ntpr", f"{self.md.nsteps} / {self.md.ntpr} → pcoord_len={self.md.pcoord_len}"))
        rows.append(("pcoord ndim", str(self.westpa.pcoord_ndim)))
        rows.append(("Walkers / bin", str(self.westpa.bin_target_counts)))
        rows.append(("Max iterations", str(self.westpa.max_iterations)))
        if self.westpa.bin_boundaries:
            nbins = len(self.westpa.bin_boundaries[0]) - 1
            rows.append(("Bins (dim 1)", f"{nbins} bins"))
        if self.method == "parGaMD":
            rows.append(("GaMD σ0D / σ0P / iE", f"{self.gamd.sigma0D} / {self.gamd.sigma0P} / {self.gamd.iE}"))
        return rows
