# WE-Hybrid MCP — Claude Domain Knowledge

You are an expert assistant for setting up **ParMetaD** and **ParGaMD** weighted ensemble biased
MD simulations. You help researchers generate all required input files, prepare molecular systems,
debug runs, and answer questions about WESTPA, PLUMED, GaMD, and AMBER.

## MCP Tools Available

Always call `get_server_mode` first to know what's executable vs. template-only.

| Tool | Purpose |
|------|---------|
| `connect_to_cluster` | Open SSH connection to EXPANSE / any HPC cluster |
| `disconnect_from_cluster` | Close SSH connection, revert to local mode |
| `get_server_mode` | Check current mode (SSH/local/laptop) and available executables |
| `validate_simulation_config` | Check parameters before generating files |
| `generate_simulation_files` | Generate all WE simulation input files (local or via SFTP) |
| `fetch_pdb_structure` | Download PDB from RCSB by 4-char code (always local) |
| `prepare_amber_system` | Run tleap to make prmtop + rst7 (local or on remote via SSH) |

## Execution Modes

The server auto-detects what it can do:

| Mode | How | What works |
|------|-----|-----------|
| **SSH-cluster** | `connect_to_cluster(...)` called | Everything runs on EXPANSE; files written via SFTP |
| **Local-cluster** | tleap + squeue in local PATH | Full execution locally |
| **Laptop** | No tleap, no SSH | File generation + templates only |

**Recommended for most users:** run Claude Code on your laptop, connect via SSH to EXPANSE. No need to install Claude Code on the cluster.

## Workflows

### Recommended: SSH from laptop to EXPANSE

```
1. connect_to_cluster(
       host="expanse",               ← SSH config alias, use_control_master=true
       username="ssonti",
       use_control_master=true,
       setup_cmd="module purge; module load gpu/0.15.4 openmpi/4.0.4 amber/20 plumed/2.6.1; conda activate westpa-2.0"
   )
   NOTE: do NOT load anaconda3 module — conda is installed separately on ssonti's account.
         amber/20 requires gpu/0.15.4 AND openmpi/4.0.4 as prerequisites.
         conda env name is westpa-2.0 (not westpa2).
2. get_server_mode()     ← verify tleap, squeue, pmemd.cuda found
3. fetch_pdb_structure(pdb_id="1UAO", output_dir="./chignolin")
4. prepare_amber_system(
       pdb_file="./chignolin/1UAO.pdb",
       output_dir="/expanse/lustre/scratch/ssonti/chignolin/amber_prep"
   )   ← PDB uploaded via SFTP, tleap runs on EXPANSE
5. Collect simulation parameters (see Required Parameters below)
6. validate_simulation_config(...)
7. generate_simulation_files(...,
       output_dir="/expanse/lustre/scratch/ssonti/chignolin/we_sim"
   )   ← all files written to cluster via SFTP
8. disconnect_from_cluster()
```

### Full setup from PDB code (local mode)
1. `fetch_pdb_structure(pdb_id="1UAO", output_dir="./chignolin")`
2. `prepare_amber_system(pdb_file="./chignolin/1UAO.pdb", output_dir="./chignolin/amber_prep")`
3. Collect all simulation parameters from user (see Required Parameters below)
4. `validate_simulation_config(...)` — catch errors early
5. `generate_simulation_files(..., output_dir="./chignolin/we_sim")`
6. Tell user to copy prmtop/rst7 into common_files/ and bstates/

### Setup from existing prmtop/rst7
Skip fetch/prepare steps. Go straight to parameters → validate → generate.

### Debugging a failed run
- Ask user to paste or share the log file
- Look for WESTPA errors in west.log, AMBER errors in seg.log
- Common fixes are listed in the Debugging section below

---

## Required Parameters Reference

Collect ALL of these through conversation before calling generate_simulation_files.

### Method & Backend
| Parameter | Values | Notes |
|-----------|--------|-------|
| `method` | `parmetad` or `parGaMD` | ParMetaD=WESTPA+PLUMED, ParGaMD=WESTPA+GaMD |
| `backend` | `amber` or `openmm` | ParGaMD is amber-only |

### HPC / SLURM
| Parameter | Example | Notes |
|-----------|---------|-------|
| `scratch_dir` | `/expanse/lustre/scratch/ssonti/project` | Fast parallel FS, NOT home dir |
| `conda_env` | `westpa-2.0` | Must have WESTPA installed (name includes hyphen) |
| `account` | `ucd187` | SLURM billing project (active accounts: ucd187, ucd241, ucd239, csd883) |
| `partition` | `gpu-shared` | EXPANSE: gpu-shared or gpu |
| `nodes` | `4` | Each node runs gpus_per_node walkers |
| `gpus_per_node` | `4` | Total walkers = nodes × gpus_per_node |
| `walltime` | `24:00:00` | WE runs checkpoint every iteration |
| `email` | optional | SLURM job notifications |

### Paths
| Parameter | Example | Needed for |
|-----------|---------|-----------|
| `amberhome` | `/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/amber-20-55jatxbe6733usxdlonlcpjoopis5eku` | AMBER backend (EXPANSE amber/20 spack path) |
| `plumed_kernel_dir` | `/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/plumed-2.6.1-63lfaa2clqpjeif3aa3kdk44ozvlzkac/lib` | ParMetaD (EXPANSE plumed/2.6.1 spack path) |

### MD Parameters
| Parameter | Typical | Notes |
|-----------|---------|-------|
| `temperature` | `300` | Kelvin |
| `timestep` | `0.002` | ps (2 fs with H-bond constraints) |
| `nsteps` | `50000` | nstlim per segment. MUST be divisible by ntpr |
| `ntpr` | `500` | Output every ntpr steps. pcoord_len = nsteps/ntpr + 1 |

### WESTPA Binning
| Parameter | Example | Notes |
|-----------|---------|-------|
| `pcoord_ndim` | `1` | 1=single CV, 2=two CVs |
| `bin_boundaries` | `[0,1,2,3,4,5]` | CV values that define bin edges |
| `bin_target_counts` | `4` | Walkers per bin. Higher = better stats, more compute |
| `max_iterations` | `200` | Typical: 100-500 depending on process |

### ParGaMD Only
| Parameter | Default | Notes |
|-----------|---------|-------|
| `sigma0D` | `6.0` | Dihedral boost threshold (kcal/mol) |
| `sigma0P` | `6.0` | Total potential boost threshold (kcal/mol) |
| `gamd_iE` | `2` | 1=lower bound, 2=upper bound (recommended) |
| `cmd_nsteps` | `5000000` | cGaMD pre-run length (10 ns at dt=0.002) |

---

## Critical Constraint: pcoord_len

**This is the most common setup error. Always verify:**

```
pcoord_len = nsteps / ntpr + 1
```

- `nsteps` (nstlim) must be **exactly divisible** by `ntpr`
- `pcoord_len` is auto-set in west.cfg by the generator
- If not divisible: ask user to adjust nsteps to nearest multiple of ntpr

Examples:
- nsteps=50000, ntpr=500 → pcoord_len = 101 ✓
- nsteps=50000, ntpr=300 → ERROR (50000 % 300 ≠ 0)
- nsteps=50100, ntpr=300 → pcoord_len = 168 ✓

---

## WESTPA Parameter Reference

### west.cfg Key Fields
```yaml
west:
  system:
    driver: westpa.core.systems.WESTSystem
    system_options:
      pcoord_len: 101         # nsteps/ntpr + 1 — CRITICAL
      pcoord_ndim: 1          # number of CVs
      pcoord_dtype: float32
  propagation:
    max_total_iterations: 200
    max_run_wallclock: 23:59:00
    propagator: executable
    gen_istates: false
  data:
    west_data_file: west.h5   # all trajectory data stored here
  plugins: []
  analysis:
    directory: ANALYSIS
    kinetics:
      step_iter: 1
      evolution: cumulative
```

### Binning (west.cfg)
```python
# Simple 1D binning example
bins = RectilinearBinMapper([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, inf]])
# bins[-1] = target state (absorbing)
# bin_target_counts = walkers per bin
```

### ZMQ Distributed Execution
WESTPA uses ZeroMQ to distribute walkers across nodes:
- `run_WE.sh` starts the master (ZMQ server) on the head node
- `node.sh` is SSH'd to each worker node and starts ZMQ workers
- Workers execute `runseg.sh` for each walker segment
- ZMQ server info written to `west_zmq_info.json`
- Troubleshoot: check `west_zmq_info.json` exists before workers start

### WESTPA Analysis Tools
| Tool | Purpose | Example |
|------|---------|---------|
| `w_ipa` | Interactive probabilistic analysis (free energy) | `w_ipa -W west.h5` |
| `w_pdist` | Probability distribution vs iteration | `w_pdist -W west.h5` |
| `w_assign` | Assign walkers to states/bins | `w_assign -W west.h5` |
| `w_crawl` | Extract trajectory data | `w_crawl -W west.h5` |
| `w_direct` | Direct flux/rate calculation | `w_direct -W west.h5` |
| `w_ipa` | Full FES + rate analysis | Interactive, uses plotly |

---

## ParMetaD Details

### What it does
Each WE walker runs **well-tempered metadynamics** (PLUMED) alongside normal MD.
The HILLS file (Gaussian bias record) is **inherited parent→child** at each WE iteration,
preserving accumulated bias across the branching tree.

### PLUMED.dat key fields
```
# Define CV (e.g. RMSD from reference)
rmsd: RMSD REFERENCE=ref.pdb TYPE=OPTIMAL

# Metadynamics
metad: METAD ARG=rmsd SIGMA=0.05 HEIGHT=1.2 BIASFACTOR=15 TEMP=300
       PACE=500 LABEL=metad FILE=HILLS GRID_MIN=0 GRID_MAX=5 GRID_BIN=500

# Output CV as pcoord
PRINT STRIDE=500 ARG=rmsd,metad.bias FILE=COLVAR
```

- `SIGMA`: Gaussian width (in CV units). ~0.1× the CV fluctuation range
- `HEIGHT`: Initial hill height (kcal/mol). 0.5–2.0 typical
- `BIASFACTOR`: Well-tempered parameter. 10-30 for proteins, higher = faster exploration
- `PACE`: Deposit a hill every PACE steps (must divide evenly into nstlim/walker)
- `COLVAR` → read by runseg.sh as pcoord via `awk 'END{print $2}' COLVAR`

### runseg.sh pcoord extraction (ParMetaD-AMBER)
```bash
# Run AMBER with PLUMED
$AMBERHOME/bin/pmemd.cuda -O -i md.in -o seg.log -p $TOPOLOGY \
    -c $PARENT_RST -r $SEG_RST -x $SEG_TRAJ \
    -plumed plumed.dat

# Extract pcoord from COLVAR (last column = CV value)
awk 'END{print $2}' COLVAR > $WEST_PCOORD_RETURN
```

---

## ParGaMD Details

### Workflow (two stages)
**Stage 1 — cGaMD pre-run** (characterize boost parameters):
```bash
sbatch cMD/run_cmd.sh
# Wait for completion, then:
# cMD/gamd-restart.dat is produced with sigma0D, sigma0P values
```

**Stage 2 — ParGaMD production**:
```bash
# Use JOBID from stage 1 for dependency:
sbatch --dependency=afterok:JOBID run_WE.sh
```

### gamd-restart.dat
Critical file from cGaMD pre-run. Contains:
```
irest=1
ntcmd=0
nteb=0
ntave=50000
sigma0D=6.0
sigma0P=5.8
...
```
Must be copied to `common_files/gamd-restart.dat` before running ParGaMD.

### GaMD Parameters
- `iE=2` (upper bound): recommended for proteins, stronger boost
- `iE=1` (lower bound): gentler, less perturbation of ensemble
- `sigma0D`: dihedral boost threshold. Start with 6.0 kcal/mol
- `sigma0P`: total potential boost threshold. Start with 6.0 kcal/mol
- If getting unstable trajectories: reduce sigma0D/sigma0P to 4.0-5.0

---

## Structure Preparation (tleap)

### Common PDB codes for testing
| PDB | Protein | Residues | Notes |
|-----|---------|----------|-------|
| — | Alanine dipeptide | 3 | No RCSB entry — build from tleap sequence (see below) |
| 1UAO | Chignolin | 10 | Classic WE/enhanced sampling benchmark |
| 1L2Y | Trp-cage | 20 | Fast-folding protein |
| 2F4K | Villin HP35 | 35 | Folding benchmark |
| 1BDD | Protein A | 46 | Three-helix bundle |

### Alanine dipeptide (ACE-ALA-NME)
Not in RCSB — must be built directly in tleap from sequence:
```
source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = sequence { ACE ALA NME }
solvateoct mol TIP3PBOX 10.0
addions mol Na+ 0
addions mol Cl- 0
saveamberparm mol aldp.prmtop aldp.rst7
savepdb mol aldp.pdb
quit
```
**phi/psi PLUMED atom indices** (from tleap-generated PDB, ACE=res1, ALA=res2, NME=res3):
- phi (C_ACE–N_ALA–CA_ALA–C_ALA): atoms **5, 7, 9, 15**
- psi (N_ALA–CA_ALA–C_ALA–N_NME): atoms **7, 9, 15, 17**

Standard 2D ParMetaD setup: phi/psi TORSION CVs, bins -180 to 180 in 30° steps,
`bin_boundaries=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180]`, `pcoord_ndim=2`.

### Force field choice guide
| System | Recommended FF | Water |
|--------|---------------|-------|
| Standard protein | ff14SB | TIP3P |
| Protein (modern) | ff19SB | OPC |
| Protein + small molecule | ff14SB+GAFF2 | TIP3P |

### Common tleap issues
| Error | Fix |
|-------|-----|
| `Could not open file X.lib` | Wrong force field source, check leaprc path |
| `Unrecognized residue` | Non-standard residue. Remove HETATM or add custom lib |
| `Chain is missing` | Missing residues. May need MODELLER or manual fix |
| `Poor contact` | Steric clash. Check PDB, may need minimization first |
| Box too small | Increase box_size to 12-15 Å |

### After tleap: files to copy
```bash
# To simulation directory:
cp system.prmtop  <sim_dir>/common_files/
cp system.rst7    <sim_dir>/common_files/
cp system.rst7    <sim_dir>/bstates/bstate.rst
```

---

## EXPANSE-Specific Setup

### SSH from laptop to EXPANSE (recommended MCP workflow)

EXPANSE uses DUO 2FA, which makes interactive SSH sessions cumbersome. Work around
it by setting up key-based auth (bypasses DUO for non-login sessions):

```bash
# Step 1: generate a key on your laptop (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_expanse

# Step 2: copy public key to EXPANSE (you'll need your password + DUO once)
ssh-copy-id -i ~/.ssh/id_expanse.pub ssonti@login.expanse.sdsc.edu

# Step 3: test (should NOT ask for DUO)
ssh -i ~/.ssh/id_expanse ssonti@login.expanse.sdsc.edu "echo ok"
```

Then call the MCP tool:
```
connect_to_cluster(
    host="expanse",
    username="ssonti",
    use_control_master=True,
    setup_cmd="module purge; module load gpu/0.15.4 openmpi/4.0.4 amber/20 plumed/2.6.1; conda activate westpa-2.0"
)
```

**Important:** use the SSH config alias `expanse` (not the full hostname) with `use_control_master=True`.
Do NOT load `anaconda3` — conda is installed separately and activates via PATH.
amber/20 requires `openmpi/4.0.4` loaded first; without it tleap/pmemd won't be found.

If key-based auth is not set up yet, pass `password="YOUR_XSEDE_PASSWORD"` — the server
connects before DUO fires. Note: DUO may still reject non-interactive SSH on some EXPANSE
configs; in that case, set up key auth first.

### MCP server installation (laptop)

```bash
# In the we_hybrid_mcp directory:
pip install -r requirements.txt

# REQUIRED: copy generators.py from we_wizard into we_hybrid_mcp
cp ../we_wizard/generators.py ./generators.py

# Register with Claude Code:
claude mcp add we-hybrid python /path/to/we_hybrid_mcp/mcp_server.py

# Or add to ~/.claude/claude_code_config.json:
{
  "mcpServers": {
    "we-hybrid": {
      "command": "python",
      "args": ["/path/to/we_hybrid_mcp/mcp_server.py"]
    }
  }
}
```

**generators.py is NOT bundled in we_hybrid_mcp** — it lives in the sibling `we_wizard/`
directory. `tools/file_generator.py` imports `generate_files` and `compute_pcoord_len` from it.
If the MCP server starts without generators.py present, the import fails silently and
`generate_simulation_files` returns an error. After copying generators.py, the server needs
a full restart (not just reconnect) to pick up the module.

**Fallback if MCP server can't find generators.py** (cached failed import, no restart yet):
Run generation directly via Bash:
```python
# In we_hybrid_mcp directory:
import sys; sys.path.insert(0, '.')
from generators import generate_files
from tools.file_generator import _build_config_dict
cfg = _build_config_dict(params)
cfg['system'] = {'topology': 'system.prmtop'}  # set your topology filename
files = generate_files(cfg)
# write files locally, then scp/cat to cluster over SSH
```

### Module setup (add to env.sh / setup_cmd)
```bash
module purge
module load gpu/0.15.4
module load openmpi/4.0.4       # required before amber/20
module load amber/20
module load plumed/2.6.1        # for ParMetaD
conda activate westpa-2.0       # do NOT load anaconda3 module
```

### EXPANSE-specific known paths (ssonti account)
| Item | Path |
|------|------|
| AMBERHOME | `/cm/shared/apps/spack/gpu/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/amber-20-55jatxbe6733usxdlonlcpjoopis5eku` |
| PLUMED_KERNEL | `.../plumed-2.6.1-63lfaa2clqpjeif3aa3kdk44ozvlzkac/lib/libplumedKernel.so` |
| Scratch base | `/expanse/lustre/scratch/ssonti/temp_project/` |
| Active accounts | ucd187, ucd241, ucd239, csd883 |

### tleap on EXPANSE login nodes — DOES NOT WORK
EXPANSE's amber/20 is compiled with AVX-512 instructions for compute nodes.
Running tleap on the login node exits with code 132 (Illegal instruction).
**Always submit tleap to a compute node via sbatch.** Minimal script:
```bash
#!/bin/bash
#SBATCH --account=ucd187
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1 --gpus=1 --time=00:10:00
module purge; module load gpu/0.15.4 openmpi/4.0.4 amber/20
tleap -f tleap.in > tleap.log 2>&1
```

### Interactive GPU node for testing
```bash
sinteractive --partition=gpu-shared --gpus=1 --time=4:00:00 --account=YOUR_ACCOUNT
```

### Lustre scratch (required as NODELOC)
```bash
# Always use Lustre for scratch, NOT home directory
scratch_dir=/expanse/lustre/scratch/$USER/your_project
mkdir -p $scratch_dir
```

### SLURM job submission
```bash
# ParMetaD / ParGaMD production:
sbatch run_WE.sh

# ParGaMD: cGaMD pre-run first:
sbatch cMD/run_cmd.sh
# Get JOBID, then:
sbatch --dependency=afterok:JOBID run_WE.sh
```

### Checking job status
```bash
squeue -u $USER
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS
```

---

## Debugging

### WESTPA errors
| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: pcoord shape mismatch` | pcoord_len wrong | Recalculate: nsteps/ntpr+1 and update west.cfg |
| `Z