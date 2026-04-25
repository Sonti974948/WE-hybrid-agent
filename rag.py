"""
rag.py — Retrieval-Augmented Generation for WESTPA domain knowledge.

Uses:
  - sentence-transformers (all-MiniLM-L6-v2) for local embeddings — no API key
  - ChromaDB for persistent vector store — saved to disk, survives sessions
  - LangChain retriever interface — plugs directly into llm_agent.py tool

Workflow:
  1. Run build_index.py ONCE on the login node to build the index
  2. At agent startup, load the saved index from disk
  3. The LLM calls retrieve_westpa_docs(query) as a tool when it needs knowledge
"""

import os
import sys
from pathlib import Path
from typing import List

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_INDEX_DIR = Path(__file__).parent / "westpa_index"
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~90 MB, fast, good for technical docs


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder(model_name: str = EMBED_MODEL):
    """Load the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB vector store helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_collection(index_dir: Path = DEFAULT_INDEX_DIR, collection_name: str = "westpa_docs"):
    """Load (or create) a persistent ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def index_documents(docs: list[dict], index_dir: Path = DEFAULT_INDEX_DIR,
                    model_name: str = EMBED_MODEL, batch_size: int = 64):
    """
    Embed and store documents in ChromaDB.

    docs: list of {"id": str, "text": str, "source": str, "title": str}
    """
    from sentence_transformers import SentenceTransformer
    import chromadb

    index_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(model_name)

    client, collection = get_chroma_collection(index_dir)

    # Check existing IDs to avoid duplicates
    existing = set(collection.get(include=[])["ids"])

    new_docs = [d for d in docs if d["id"] not in existing]
    if not new_docs:
        print(f"  All {len(docs)} documents already indexed.")
        return

    print(f"  Embedding {len(new_docs)} new documents (batch_size={batch_size})...")

    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i:i + batch_size]
        texts = [d["text"] for d in batch]
        embeddings = embedder.encode(texts, show_progress_bar=True,
                                     device="cuda" if _cuda_available() else "cpu")
        collection.add(
            ids=[d["id"] for d in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": d.get("source", ""), "title": d.get("title", "")} for d in batch],
        )
        print(f"  Indexed {min(i + batch_size, len(new_docs))}/{len(new_docs)}")

    print(f"  ✓ Index now contains {collection.count()} documents.")


def retrieve(query: str, k: int = 5, index_dir: Path = DEFAULT_INDEX_DIR,
             model_name: str = EMBED_MODEL) -> list[dict]:
    """
    Retrieve top-k most relevant document chunks for a query.
    Returns list of {"text": str, "source": str, "title": str, "score": float}
    """
    embedder = get_embedder(model_name)
    _, collection = get_chroma_collection(index_dir)

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "score": round(1 - dist, 3),   # cosine distance → similarity
        })
    return chunks


def format_retrieved_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a string for injection into the LLM context."""
    if not chunks:
        return "No relevant documentation found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['title']} (source: {chunk['source']}, relevance: {chunk['score']})"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def index_ready(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    """Check if the ChromaDB index exists and has documents."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection("westpa_docs")
        return collection.count() > 0
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Built-in expert knowledge documents
# Written specifically for this wizard — covers what Llama 3.2 may not know
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_DOCS = [
    {
        "id": "westpa_overview",
        "title": "WESTPA Overview and Architecture",
        "source": "expert_knowledge",
        "text": """
WESTPA (Weighted Ensemble Simulation Toolkit with Parallelization and Analysis) implements
the Weighted Ensemble (WE) method for sampling rare events in molecular simulations.

Core concept: Instead of running one long simulation, WE runs many short parallel
trajectory segments (walkers) and uses statistical resampling (splitting/merging) to
maintain walkers across all regions of conformational space.

Key components:
- west.cfg: Master YAML configuration file
- west.h5: HDF5 data file storing all trajectory data, weights, pcoords
- westpa_scripts/: Shell scripts called by WESTPA for each segment
- traj_segs/: Directory storing trajectory data per iteration
- seg_logs/: Log files per segment
- istates/: Initial states for new trajectories

WESTPA 2.0 improvements over 1.0:
- ZMQ work manager for distributed GPU computing (replaces MPI)
- Better checkpoint/restart support
- Improved analysis tools (w_ipa interactive analysis)
- Support for adaptive binning schemes
- Better memory management for large simulations
""",
    },
    {
        "id": "west_cfg_schema",
        "title": "west.cfg Complete Configuration Reference",
        "source": "expert_knowledge",
        "text": """
west.cfg is the master WESTPA configuration file (YAML format).

Full schema with explanations:

west:
  system:
    driver: westpa.core.systems.WESTSystem
    system_options:
      pcoord_ndim: 1          # Number of progress coordinate dimensions (1 or 2 typically)
      pcoord_len: 101         # Data points per iteration = nstlim/ntpr + 1 (CRITICAL CONSTRAINT)
      pcoord_dtype: !!python/name:numpy.float32
      bins:
        type: RectilinearBinMapper
        boundaries:
          - [0.0, 1.0, 2.0, ..., 'inf']   # Bin edges for dim 1; 'inf' catches out-of-range walkers
          - [0.0, 1.0, 2.0, ..., 'inf']   # Bin edges for dim 2 (if pcoord_ndim=2)
      bin_target_counts: 4    # Target walkers per bin; WESTPA splits/merges to maintain this

  propagation:
    max_total_iterations: 200   # Maximum WE iterations to run
    max_run_wallclock: 47:30:00 # Stop before this walltime (safety margin)
    propagator: executable      # Use shell script propagator (vs. Python API)
    gen_istates: false          # false = use provided basis states; true = generate new ones

  data:
    west_data_file: west.h5     # All simulation data stored here (HDF5)
    datasets:
      - name: pcoord
        scaleoffset: 4          # HDF5 compression precision
      - name: coord
        dtype: float32
        scaleoffset: 3
    data_refs:
      segment: $WEST_SIM_ROOT/traj_segs/{segment.n_iter:06d}/{segment.seg_id:06d}
      basis_state: $WEST_SIM_ROOT/bstates/{basis_state.auxref}
      initial_state: $WEST_SIM_ROOT/istates/{initial_state.iter_created}/{initial_state.state_id}.rst

  executable:
    propagator:
      executable: $WEST_SIM_ROOT/westpa_scripts/runseg.sh    # Called for each walker each iteration
      stdout: $WEST_SIM_ROOT/seg_logs/{segment.n_iter:06d}-{segment.seg_id:06d}.log
      stderr: stdout
    get_pcoord:
      executable: $WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh  # Called for basis/initial states
    gen_istate:
      executable: $WEST_SIM_ROOT/westpa_scripts/gen_istate.sh
    post_iteration:
      enabled: true
      executable: $WEST_SIM_ROOT/westpa_scripts/post_iter.sh   # Called after each full iteration
    pre_iteration:
      enabled: false
      executable: $WEST_SIM_ROOT/westpa_scripts/pre_iter.sh

scaleoffset: Controls HDF5 lossy compression. Higher = more compression, less precision.
  scaleoffset: 4 means 4 significant figures retained.
""",
    },
    {
        "id": "pcoord_bins_strategy",
        "title": "Progress Coordinate and Bin Strategy",
        "source": "expert_knowledge",
        "text": """
The progress coordinate (pcoord) is what WESTPA uses to define bins and resample walkers.

Choosing a good pcoord:
- Must be computable from each segment's output files
- Should discriminate between important states (folded/unfolded, bound/unbound)
- Common choices:
  * CA-RMSD from reference structure (good for folding)
  * Radius of gyration (compactness)
  * Inter-residue Calpha distance (binding)
  * Number of native contacts
  * Combination of 2 CVs (2D pcoord)

Setting bin boundaries:
- Cover the FULL expected range of your pcoord
- Always include 'inf' as the last boundary to catch out-of-range walkers
- Too few bins: poor sampling coverage, walkers bunch up
- Too many bins: each bin has too few walkers, poor statistics
- Typical: 20-50 bins for 1D, 15x15 to 30x30 grid for 2D
- Make bins finer in regions of interest (transition states)

bin_target_counts:
- How many walkers WESTPA tries to maintain per bin
- More walkers = better statistics, more GPU time
- 2-4 walkers/bin: memory-efficient, good for exploration
- 8-16 walkers/bin: better statistics, more expensive
- Total walkers ≈ n_bins × bin_target_counts (varies as bins fill)

pcoord_len CONSTRAINT (critical):
- pcoord_len = nstlim / ntpr + 1
- This MUST be exact — WESTPA will crash with a shape mismatch otherwise
- Example: nstlim=50000, ntpr=500 → pcoord_len=101
- The +1 accounts for the parent frame at the start of each segment

WESTPA environment variables available in scripts:
- $WEST_SIM_ROOT: simulation root directory
- $WEST_CURRENT_SEG_DATA_REF: path to current segment data
- $WEST_PARENT_DATA_REF: path to parent segment data
- $WEST_PCOORD_RETURN: file to write pcoord data to
- $WEST_CURRENT_SEG_INITPOINT_TYPE: SEG_INITPOINT_CONTINUES or SEG_INITPOINT_NEWTRAJ
- $WEST_RAND16: 16-character random string for random seeds
- $WM_PROCESS_INDEX: worker index (use for GPU assignment)
- $CUDA_VISIBLE_DEVICES_ALLOCATED: comma-separated list of GPUs for this worker
""",
    },
    {
        "id": "zmq_distributed",
        "title": "ZMQ Distributed Execution on GPU Clusters",
        "source": "expert_knowledge",
        "text": """
WESTPA 2.0 uses ZeroMQ (ZMQ) for distributed computing across multiple GPU nodes.

Architecture:
- Master process: runs on the head node of the SLURM job, coordinates iterations
- Worker processes: run on each compute node, execute the actual MD segments

How it works:
1. run_WE.sh starts the ZMQ master with: w_run --work-manager=zmq --zmq-mode=master
2. Master writes its address to west_zmq_info.json
3. node.sh is SSH'd to each compute node and starts workers: w_run --zmq-mode=client
4. Workers connect to master, receive segment tasks, run MD, return pcoord data

Key environment variables for ZMQ:
- WM_ZMQ_MASTER_HEARTBEAT: 100 (seconds between master heartbeats)
- WM_ZMQ_WORKER_HEARTBEAT: 100 (seconds between worker heartbeats)
- WM_ZMQ_TIMEOUT_FACTOR: 300 (worker timeout multiplier)
- SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info.json

GPU assignment per walker:
- CUDA_VISIBLE_DEVICES_ALLOCATED: all GPUs allocated to this node (e.g., "0,1,2,3")
- WM_PROCESS_INDEX: index of this worker (0, 1, 2, ...)
- In runseg.sh: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$WM_PROCESS_INDEX]}
- This assigns one GPU per walker

SLURM job structure for multi-node WE:
- --nodes=N: number of nodes
- --gpus=N*gpus_per_node: total GPUs
- --ntasks-per-node=1: one task per node (the ZMQ worker launcher)

Common ZMQ errors:
- "server failed to start": check west-JOBID-local.log, usually a Python/conda issue
- Workers can't connect: check firewall, StrictHostKeyChecking, SSH keys between nodes
- Timeout: increase WM_ZMQ_TIMEOUT_FACTOR or reduce segment length

nodefilelist.txt:
- Created by: scontrol show hostname $SLURM_JOB_NODELIST > nodefilelist.txt
- Contains one hostname per line for all nodes in the job
- node.sh iterates over this to start workers on each node
""",
    },
    {
        "id": "westpa_analysis_tools",
        "title": "WESTPA Analysis Tools Reference",
        "source": "expert_knowledge",
        "text": """
WESTPA provides several command-line tools for analyzing west.h5 data.

w_ipa (Interactive Progress Analysis) — most commonly used:
  w_ipa -r west.h5 --force
  Interactive Python session with access to all simulation data.
  Key commands inside w_ipa:
    west.data_reader.load_iter_data(n)  # load iteration n
    west.plot_flux()                     # plot flux into target state

w_pdist (Probability Distribution):
  w_pdist -r west.h5 -o pdist.h5 --first-iter 10 --last-iter 50
  Computes probability distribution along pcoord.
  Options:
    --first-iter N: skip first N iterations (equilibration)
    --last-iter N: use only up to iteration N
    --bins 50: number of histogram bins

w_assign (State Assignment):
  w_assign -r west.h5 -s states.yaml -o assign.h5
  Assigns trajectory segments to macrostates.
  Required: states.yaml defining state boundaries.

w_fluxanl (Flux Analysis):
  w_fluxanl -r west.h5 -a assign.h5 -o fluxanl.h5
  Computes steady-state fluxes between states.

w_stateprobs (State Probabilities):
  w_stateprobs -r west.h5 -a assign.h5 -o stateprobs.h5

w_kinetics (Rate Constants):
  w_kinetics -r west.h5 -a assign.h5 -o kinetics.h5
  Computes rate constants from flux data.

w_direct (Direct Rate Estimation):
  w_direct -r west.h5 -a assign.h5 -o direct.h5
  Direct rate constant estimation without flux analysis.

w_truncate (Truncate simulation):
  w_truncate -r west.h5 -n 10
  Truncates simulation to first 10 iterations (for restarts).

Checking simulation progress:
  h5ls -r west.h5                     # list all datasets
  python -c "import h5py; f=h5py.File('west.h5'); print(f['iterations'].keys())"

Restarting a crashed WE simulation:
  # No need to re-init — just resubmit run_WE.sh without calling ./init.sh
  # WESTPA reads west.h5 and continues from the last completed iteration
  # If last iteration is corrupted: w_truncate -n LAST_GOOD_ITER
""",
    },
    {
        "id": "gamd_parameters",
        "title": "GaMD Parameters and ParGaMD Workflow",
        "source": "expert_knowledge",
        "text": """
Gaussian Accelerated Molecular Dynamics (GaMD) adds a harmonic boost potential to
overcome energy barriers without requiring predefined reaction coordinates.

GaMD AMBER input parameters:
  igamd = 3          # Dual boost (dihedral + total potential); igamd=1 (total only), igamd=2 (dihedral only)
  iE = 2             # Upper threshold energy (recommended); iE=1 uses lower threshold
  irest_gamd = 0     # 0 = fresh GaMD run (cGaMD); 1 = restart with saved parameters
  ntcmd = 1000000    # cGaMD equilibration steps (collect statistics for boost params)
  nteb = 2000000     # Total cGaMD steps including boost equilibration
  ntave = 50000      # Averaging window for boost parameter update
  ntcmdprep = 200000 # Steps before collecting boost statistics
  ntebprep = 200000  # Boost equilibration preparation steps
  sigma0D = 6.0      # Upper bound std dev of dihedral boost (kcal/mol)
  sigma0P = 6.0      # Upper bound std dev of total potential boost (kcal/mol)

Choosing sigma0:
  - sigma0D = sigma0P = 6.0 kcal/mol: standard for small proteins in implicit solvent
  - For explicit solvent: sigma0P may need to be larger (10-20 kcal/mol)
  - Lower sigma0: gentler boost, less perturbation, better accuracy but slower sampling
  - Higher sigma0: stronger boost, faster sampling but may cause instability

ParGaMD workflow:
  Step 1: cGaMD pre-run (generates gamd-restart.dat)
    - Run conventional GaMD to collect boost statistics
    - Output: gamd.log contains gamd-restart.dat parameters
    - Typical: 1-2 million steps
    - Submit: sbatch cMD/run_cmd.sh
    - Wait for completion: squeue -u $USER

  Step 2: WE production run (uses gamd-restart.dat)
    - irest_gamd = 1 in md.in reads parameters from gamd-restart.dat
    - Submit as dependency: sbatch --dependency=afterok:JOBID run_WE.sh
    - gamd-restart.dat is copied to each walker segment directory

  Extracting FES from ParGaMD:
    1. Get weights: awk 'NR%1==0' gamd.log | awk '{print ($8+$7)/(0.001987*300)" "$2" "($8+$7)}' > weights.dat
    2. Get pcoords: awk 'NR==FNR{a[NR]=$2; next} {print a[FNR], $2}' PC1.dat PC2.dat > output.dat
    3. Reweight: python PyReweighting-2D.py -input output.dat -Xmax 8 -Ymax 8 -discX 0.1 -discY 0.1 -T 300 -Emax 20 -job reweight_ME -weight weights.dat
    4. Output: pmf-c2-ME.dat contains the free energy surface

gamd.log columns (AMBER):
  Col 1: step, Col 2: Vmax, Col 3: Vmin, Col 4: Vavg, Col 5: sigmaV
  Col 6: Vdmax, Col 7: dV (dihedral boost), Col 8: dVp (total potential boost)
""",
    },
    {
        "id": "metadynamics_plumed",
        "title": "PLUMED Metadynamics Configuration for ParMetaD",
        "source": "expert_knowledge",
        "text": """
PLUMED is used in ParMetaD to add well-tempered metadynamics bias to each WE walker.

plumed.dat structure:
  RESTART                          # Required for continuing from parent HILLS

  # Collective Variable definitions
  rmsd: RMSD REFERENCE=ref.pdb TYPE=OPTIMAL   # RMSD from reference (Angstroms)
  rg: GYRATION TYPE=RADIUS ATOMS=@CA          # Radius of gyration

  # Well-Tempered MetaDynamics
  metad: METAD ARG=rmsd SIGMA=0.05 HEIGHT=1.2 PACE=500 BIASFACTOR=10 TEMP=300 FILE=HILLS RESTART=YES

  PRINT ARG=rmsd,metad.bias STRIDE=100 FILE=COLVAR

Key METAD parameters:
  ARG: which CV(s) to bias (comma-separated for 2D)
  SIGMA: Gaussian width in CV units — too small = slow filling, too large = coarse FES
    * For RMSD: 0.05-0.1 Angstroms typical
    * For distance: 0.1-0.2 Angstroms typical
    * Rule of thumb: ~1/10 of the CV range you want to resolve
  HEIGHT: initial Gaussian height (kJ/mol in PLUMED) — controls boost strength
  PACE: deposit a Gaussian every PACE steps
  BIASFACTOR: well-tempering parameter (γ). Higher = more aggressive flattening
    * 5-15 typical for proteins; 10 is a common starting point
  TEMP: system temperature in Kelvin
  FILE=HILLS: output file for Gaussian kernels (inherited by child walkers)
  RESTART=YES: required to continue accumulating bias from parent HILLS

plumed_init.dat: same as plumed.dat but WITHOUT RESTART keyword and RESTART=YES
  Used only for the very first iteration (SEG_INITPOINT_NEWTRAJ)

HILLS file management in ParMetaD:
  In runseg.sh:
    if SEG_INITPOINT_CONTINUES: cp $WEST_PARENT_DATA_REF/HILLS HILLS
    if SEG_INITPOINT_NEWTRAJ:   touch HILLS   (empty for fresh start)
  This ensures the accumulated bias is passed from parent to child walkers.

COLVAR to pcoord:
  The COLVAR file contains CV values at each STRIDE.
  runseg.sh extracts: tail -n +2 COLVAR | awk '{print $2}' > $WEST_PCOORD_RETURN
  For 2D: paste <(awk '{print $2}') <(awk '{print $3}') > $WEST_PCOORD_RETURN

Common PLUMED errors:
  "PLUMED: HILLS file not found": check HILLS is being copied in runseg.sh
  "PLUMED: wrong number of arguments": check ARG matches number of SIGMA values
  "libplumedKernel.so not found": set PLUMED_KERNEL env variable in env.sh
  Segfault on AMBER+PLUMED: make sure PLUMED was compiled against the same AMBER version
""",
    },
    {
        "id": "common_errors",
        "title": "Common WESTPA Errors and Solutions",
        "source": "expert_knowledge",
        "text": """
Common errors and how to fix them:

1. "ValueError: could not broadcast input array from shape (X,) into shape (Y,)"
   Cause: pcoord_len in west.cfg doesn't match nstlim/ntpr + 1
   Fix: set pcoord_len = nstlim / ntpr + 1 exactly

2. "ZMQ master failed to start"
   Cause: WESTPA/Python environment not loaded properly on master node
   Fix: check west-JOBID-local.log; ensure conda activate runs before w_run

3. "Workers can't connect to master"
   Cause: SSH between compute nodes blocked, or west_zmq_info.json not written
   Fix: test ssh node1 from node2; check WEST_SIM_ROOT is accessible on all nodes

4. "FileNotFoundError: bstates/bstate.rst"
   Cause: basis state file missing or wrong path in bstates.txt
   Fix: check bstates/bstates.txt format: "0 1 filename.rst"

5. "Segmentation fault in pmemd.cuda"
   Cause: GPU memory issue, corrupt restart file, or bad AMBER input
   Fix: check seg.log for AMBER error; try with one GPU first

6. "PLUMED: internal error"
   Cause: HILLS file from different CV definition, or PLUMED version mismatch
   Fix: delete HILLS and restart; check PLUMED_KERNEL version

7. "west.h5 file is corrupt"
   Cause: job killed mid-write
   Fix: w_truncate -r west.h5 -n LAST_GOOD_ITER; resubmit without init.sh

8. Simulation runs but pcoord stays constant
   Cause: get_pcoord.sh or runseg.sh not writing to $WEST_PCOORD_RETURN
   Fix: add "set -x" to runseg.sh for debugging; check COLVAR is being produced

9. "No space left on device" on NODELOC
   Cause: per-walker temp files filling scratch filesystem
   Fix: enable tar_segs.sh in post_iter.sh; increase scratch allocation

10. All walkers pile up in one bin
    Cause: bins too coarse, or pcoord not discriminating well
    Fix: use finer bins near expected transition; check pcoord computation

Debugging tips:
  - Add SEG_DEBUG=1 to segment environment for verbose output
  - Check seg_logs/000001-000000.log for first iteration errors
  - Run w_init manually before submitting run_WE.sh to catch config errors
  - Use --work-manager=threads for single-node debugging (no ZMQ needed)
""",
    },
    {
        "id": "amber_md_input",
        "title": "AMBER MD Input File Parameters for WE Simulations",
        "source": "expert_knowledge",
        "text": """
AMBER .in file parameters relevant to WE simulations:

Basic control:
  irest = 0/1     # 0 = new run (no velocities), 1 = restart (read velocities)
  ntx = 1/5       # 1 = read coordinates only, 5 = read coordinates + velocities
  nstlim = 50000  # number of MD steps per WE segment
  dt = 0.002      # timestep in ps (2 fs with SHAKE, 4 fs with HMR)

Output (critical for pcoord_len):
  ntpr = 500      # write energy to log every ntpr steps
  ntwx = 500      # write coordinates to trajectory every ntpr steps
  ntwr = 500      # write restart file every ntpr steps
  ioutfm = 1      # 1 = binary NetCDF trajectory (recommended)
  ntxo = 2        # 2 = binary restart file

pcoord_len = nstlim/ntpr + 1 = 50000/500 + 1 = 101

Temperature control:
  ntt = 3         # Langevin thermostat (recommended for WE)
  temp0 = 300     # target temperature (K)
  gamma_ln = 5    # Langevin collision frequency (1/ps); 1-5 typical
  ig = RAND       # random seed (WESTPA replaces RAND with $WEST_RAND16)
  ig = -1         # alternative: use system time (not reproducible)

Constraints:
  ntc = 2         # SHAKE on H-bonds (allows 2 fs timestep)
  ntf = 2         # skip H-bond force evaluation

Implicit solvent (GB):
  igb = 1         # HCT model (fast, good for small proteins)
  igb = 2/5/8     # OBC1/OBC2/GBn2 (more accurate but slower)
  cut = 9999.0    # no cutoff for implicit solvent
  rgbmax = 9999.0 # no cutoff for Born radii

Explicit solvent:
  igb = 0         # no GB (default)
  cut = 9.0       # standard 9 Å cutoff for explicit
  ntb = 1         # periodic boundaries (1=constant V, 2=constant P)

Hydrogen Mass Repartitioning (HMR) for 4 fs timestep:
  dt = 0.004      # 4 fs timestep
  ntc = 2         # still SHAKE on H-bonds
  # Requires topology modified with: parmed topology.prmtop HMassRepartition

Random seed for reproducibility in WE:
  ig = RAND in md.in → sed "s/RAND/$WEST_RAND16/g" in runseg.sh
  Each walker gets a unique 16-char hex seed ensuring different trajectories
""",
    },
    {
        "id": "bstates_istates",
        "title": "Basis States and Initial States in WESTPA",
        "source": "expert_knowledge",
        "text": """
WESTPA uses two types of states to initialize simulations:

Basis States (bstates):
- The starting structures for the simulation
- Defined in bstates/bstates.txt: "probability label auxref"
- Example: "0 1 bstate.rst" (probability=0, label=1, file=bstate.rst)
- Multiple basis states possible for multiple starting conformations
- For ParGaMD: bstate.rst should be a structure after equilibration
- For ParMetaD: bstate.rst (AMBER) or model.xyz (GPUMD)
- get_pcoord.sh is called for each basis state to compute initial pcoord

bstates.txt format:
  0 1 bstate.rst           # single basis state
  0.5 1 state1.rst         # two equal-probability basis states
  0.5 2 state2.rst

Initial States (istates):
- Generated by gen_istate.sh from basis states
- Stored in istates/ITER/STATE_ID.rst
- When gen_istates: false in west.cfg, basis states ARE initial states
- When gen_istates: true, gen_istate.sh is called to create new starts

Target States (tstate):
- Optional: defines when a WE simulation has "succeeded"
- Defined in tstate.file: "label pcoord_value"
- Example: "folded 1.5" (RMSD < 1.5 Å = folded state)
- WESTPA stops recycling walkers that reach the target
- Only needed for rate constant calculations, not required for FES

init.sh:
  Calls w_init to set up the simulation directory
  Key flags:
    --bstate-file bstates/bstates.txt
    --tstate-file tstate.file (optional)
    --segs-per-state N: initial walkers per basis state
  Creates: traj_segs/, seg_logs/, istates/, west.h5
  Run ONCE before the first WE run; NOT needed for restarts
""",
    },
    {
        "id": "expanse_specific",
        "title": "EXPANSE HPC Cluster Specific Configuration",
        "source": "expert_knowledge",
        "text": """
EXPANSE (San Diego Supercomputer Center) specific settings for WE simulations:

Filesystem layout:
  Home: /home/username (limited space, ~100 GB, NFS)
  Scratch: /expanse/lustre/scratch/username/PROJECT (fast Lustre, large, temporary)
  Project: /expanse/projects/PROJECT (persistent, shared)

Always set NODELOC to scratch: /expanse/lustre/scratch/username/project_name

Useful SLURM partitions on EXPANSE:
  gpu-shared: 1 GPU per job, shared node, good for testing
  gpu: full GPU nodes (4 GPUs per node), good for production WE
  gpu-debug: short debug runs (<30 min)

Module setup for WESTPA on EXPANSE:
  module purge
  module load shared
  module load gpu/0.15.4
  module load slurm
  module load openmpi/4.0.4
  module load cuda/11.0.2
  module load amber/20-patch15   # if using AMBER backend
  conda activate westpa-2.0

EXPANSE GPU nodes:
  4 NVIDIA V100 (32 GB) GPUs per node
  GPU partition: up to 4 nodes (16 GPUs total) per job
  GPU-shared: 1 GPU from a shared node

Compute node internet access:
  Login nodes: have internet access (download Ollama, pip install, etc.)
  Compute nodes: NO internet access (use modules, conda, pre-downloaded models)
  Solution: install everything on login node, models cached in ~/.ollama/

SSH between compute nodes:
  EXPANSE allows SSH between nodes in the same job (required for ZMQ node.sh)
  StrictHostKeyChecking=no is needed in node.sh for automated SSH

Typical SLURM job for WE on EXPANSE:
  #SBATCH --partition=gpu
  #SBATCH --nodes=4
  #SBATCH --gpus=16          # 4 nodes × 4 GPUs
  #SBATCH --ntasks-per-node=1
  #SBATCH --account=YOUR_ACCOUNT
  #SBATCH -t 24:00:00

Checking GPU availability: sinfo -p gpu --Format=nodes,cpus,memory,gres
Checking account balance: expanse-client user -p YOUR_ACCOUNT
""",
    },
]


def get_expert_docs() -> list[dict]:
    """Return the built-in expert knowledge documents."""
    return EXPERT_DOCS
