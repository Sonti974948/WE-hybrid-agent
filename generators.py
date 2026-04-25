"""
generators.py
File generation logic for WE-Hybrid simulations.
Supports: ParMetaD (AMBER, OpenMM) and ParGaMD (AMBER)
"""

import zipfile
import io


def compute_pcoord_len(nsteps, ntpr):
    """Enforce: pcoord_len = nstlim / ntpr + 1"""
    return int(int(nsteps) / int(ntpr)) + 1


def fmt_bins(boundaries_list):
    """Format bin boundaries list for west.cfg YAML"""
    lines = []
    for b in boundaries_list:
        parts = ", ".join(
            f"'{v}'" if isinstance(v, str) else str(v) for v in b
        )
        lines.append(f"          - [{parts}]")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FILES
# ─────────────────────────────────────────────────────────────────────────────

def gen_bstates_txt(config):
    method = config["method"]
    backend = config["backend"]
    if method == "parmetad" and backend == "openmm":
        return "0 1 bstate.inpcrd\n"
    elif method == "parmetad" and backend == "amber":
        return "0 1 bstate.rst\n"
    else:  # parGaMD
        return "0 1 bstate.rst\n"


def gen_init_sh(config):
    tstate = config.get("tstate", False)
    tstate_line = 'TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"' if tstate else '#TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"'
    return f"""#!/bin/bash

# Set up simulation environment
source env.sh

# Clean up from previous / failed runs
rm -rf traj_segs seg_logs istates west.h5
mkdir   seg_logs traj_segs istates

# Set pointer to bstate and tstate
BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/bstates/bstates.txt"
{tstate_line}

# Run w_init
w_init \\
  $BSTATE_ARGS \\
  $TSTATE_ARGS \\
  --segs-per-state {config['westpa']['segs_per_state']} \\
  --work-manager=threads "$@"
"""


def gen_node_sh(config):
    return """#!/bin/bash
# node.sh — launched via SSH on each compute node

set -x
umask g+r
cd $1; shift
source env.sh
export WEST_JOBID=$1; shift
export SLURM_NODENAME=$1; shift
export CUDA_VISIBLE_DEVICES_ALLOCATED=$1; shift
echo "starting WEST client processes on: "; hostname
echo "current directory is $PWD"
env | sort

w_run "$@" &> west-$SLURM_NODENAME-node.log
echo "Shutting down.  Hopefully this was on purpose?"
"""


def gen_post_iter_sh():
    return """#!/bin/bash
# post_iter.sh — called after each WE iteration

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
"""


def gen_gen_istate_sh():
    return """#!/bin/bash
# gen_istate.sh — generates initial states

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
"""


def gen_tar_segs_sh():
    return """#!/bin/bash
# tar_segs.sh — compress segment data to save disk space
# Usage: bash tar_segs.sh <iter_number>

ITER=$1
ITER_DIR="traj_segs/$(printf '%06d' $ITER)"
if [ -d "$ITER_DIR" ]; then
    tar -czf "${ITER_DIR}.tar.gz" "$ITER_DIR" && rm -rf "$ITER_DIR"
    echo "Compressed $ITER_DIR"
fi
"""


def gen_cat_trajectory_py():
    return '''#!/usr/bin/env python3
"""
cat_trajectory.py — concatenate trajectory segments for a given iteration.
Usage: python cat_trajectory.py <iter_number>
"""
import sys
import os
import glob

def main():
    if len(sys.argv) < 2:
        print("Usage: cat_trajectory.py <iter>")
        sys.exit(1)

    iteration = int(sys.argv[1])
    iter_dir = f"traj_segs/{iteration:06d}"

    if not os.path.exists(iter_dir):
        print(f"Iteration directory not found: {iter_dir}")
        sys.exit(1)

    segs = sorted(glob.glob(f"{iter_dir}/*/"))
    print(f"Found {len(segs)} segments in iteration {iteration}")
    for seg in segs:
        print(seg)

if __name__ == "__main__":
    main()
'''


# ─────────────────────────────────────────────────────────────────────────────
# AMBER FILES (shared between ParMetaD-AMBER and ParGaMD)
# ─────────────────────────────────────────────────────────────────────────────

def gen_env_sh_amber(config):
    hpc = config["hpc"]
    paths = config["paths"]
    method = config["method"]

    amber_mod = f"module load amber/{hpc.get('amber_version', '20-patch15')}"
    plumed_export = ""
    if method == "parmetad":
        plumed_export = f"\nexport LD_LIBRARY_PATH={paths['plumed_kernel_dir']}:$LD_LIBRARY_PATH"
        plumed_export += f"\nexport PLUMED_KERNEL={paths['plumed_kernel_dir']}/libplumedKernel.so"

    return f"""#!/bin/bash

source ~/.bash_profile
module purge
module load shared
module load gpu/0.15.4
module load slurm
module load openmpi/4.0.4
module load cuda/{hpc.get('cuda_version', '11.0.2')}
{amber_mod}
conda activate {hpc['conda_env']}

export PATH=$PATH:$HOME/bin
export PYTHONPATH=$(which python){plumed_export}

# Simulation root directory
if [[ -z "$WEST_SIM_ROOT" ]]; then
    export WEST_SIM_ROOT="$PWD"
fi

export SIM_NAME=$(basename $WEST_SIM_ROOT)
echo "simulation $SIM_NAME root is $WEST_SIM_ROOT"

source $AMBERHOME/amber.sh

# Node-local scratch — use a fast filesystem
export NODELOC={hpc['scratch_dir']}
export USE_LOCAL_SCRATCH=1

# ZMQ settings
export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT_FACTOR=300

# System utilities (explicit paths for cluster reliability)
export BASH=$SWROOT/bin/bash
export LN=$SWROOT/bin/ln
export CP=$SWROOT/bin/cp
export RM=$SWROOT/bin/rm
export SED=$SWROOT/bin/sed
export CAT=$SWROOT/bin/cat
export TAR=$SWROOT/bin/tar
export AWK=$SWROOT/usr/bin/awk
export PASTE=$SWROOT/usr/bin/paste
export GREP=$SWROOT/bin/grep
export SORT=$SWROOT/usr/bin/sort
export MKDIR=$SWROOT/bin/mkdir

# AMBER executables
export SANDER=$AMBERHOME/bin/sander
export PMEMD=$AMBERHOME/bin/pmemd.cuda
export CPPTRAJ=$AMBERHOME/bin/cpptraj
"""


def gen_run_we_sh_amber(config):
    hpc = config["hpc"]
    method = config["method"]
    job_name = "ParMetaD_WE" if method == "parmetad" else "ParGaMD_WE"
    plumed_line = ""
    if method == "parmetad":
        paths = config["paths"]
        plumed_line = f"\nexport LD_LIBRARY_PATH={paths['plumed_kernel_dir']}:$LD_LIBRARY_PATH"

    amber_mod = f"module load amber/{hpc.get('amber_version', '20-patch15')}"
    email_line = f"#SBATCH --mail-user={hpc['email']}" if hpc.get('email') else "#SBATCH --mail-user=your@email.com"

    return f"""#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --output="job.out"
#SBATCH --error="job.err"
#SBATCH --partition={hpc['partition']}
#SBATCH --nodes={hpc['nodes']}
#SBATCH --gpus={int(hpc['nodes']) * int(hpc['gpus_per_node'])}
#SBATCH --ntasks-per-node=1
#SBATCH --mem={hpc.get('mem', '50G')}
#SBATCH --account={hpc['account']}
#SBATCH --no-requeue
{email_line}
#SBATCH --mail-type=ALL
#SBATCH -t {hpc['walltime']}

set -x
cd $SLURM_SUBMIT_DIR
source ~/.bashrc
module purge
module load shared
module load gpu/0.15.4
module load slurm
module load openmpi/4.0.4
module load cuda/{hpc.get('cuda_version', '11.0.2')}
{amber_mod}
conda activate {hpc['conda_env']}{plumed_line}

export WEST_SIM_ROOT=$SLURM_SUBMIT_DIR
cd $WEST_SIM_ROOT

./init.sh
echo "init.sh ran"
source env.sh || exit 1

SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info.json

num_gpu_per_node={hpc['gpus_per_node']}
rm -rf nodefilelist.txt
scontrol show hostname $SLURM_JOB_NODELIST > nodefilelist.txt

# Start ZMQ master
w_run --work-manager=zmq --n-workers=0 \\
      --zmq-mode=master \\
      --zmq-write-host-info=$SERVER_INFO \\
      --zmq-comm-mode=tcp &> west-$SLURM_JOBID-local.log &

# Wait for master to start (up to 60 seconds)
for ((n=0; n<60; n++)); do
    if [ -e $SERVER_INFO ] ; then
        echo "== server info file $SERVER_INFO =="
        cat $SERVER_INFO
        break
    fi
    sleep 1
done

if ! [ -e $SERVER_INFO ] ; then
    echo 'ZMQ master failed to start — check west-*-local.log'
    exit 1
fi

# Start workers on each node
CUDA_LIST=$(seq -s, 0 $((num_gpu_per_node - 1)))
export CUDA_VISIBLE_DEVICES=$CUDA_LIST

for node in $(cat nodefilelist.txt); do
    ssh -o StrictHostKeyChecking=no $node \\
        $PWD/node.sh $SLURM_SUBMIT_DIR $SLURM_JOBID $node $CUDA_VISIBLE_DEVICES \\
        --work-manager=zmq --n-workers=$num_gpu_per_node \\
        --zmq-mode=client \\
        --zmq-read-host-info=$SERVER_INFO \\
        --zmq-comm-mode=tcp &
done
wait
"""


def gen_west_cfg(config):
    westpa = config["westpa"]
    md = config["md"]
    pcoord_len = compute_pcoord_len(md["nsteps"], md["ntpr"])
    bins_yaml = fmt_bins(westpa["bin_boundaries"])

    return f"""# WEST configuration file — auto-generated by WE-Hybrid Setup Wizard
# vi: set filetype=yaml :
---
west:
  system:
    driver: westpa.core.systems.WESTSystem
    system_options:
      # Dimensionality of your progress coordinate
      pcoord_ndim: {westpa['pcoord_ndim']}
      # pcoord_len = nstlim/ntpr + 1 = {md['nsteps']}/{md['ntpr']} + 1 = {pcoord_len}
      pcoord_len: {pcoord_len}
      pcoord_dtype: !!python/name:numpy.float32
      bins:
        type: RectilinearBinMapper
        boundaries:
{bins_yaml}
      # Walkers per bin
      bin_target_counts: {westpa['bin_target_counts']}
  propagation:
    max_total_iterations: {westpa['max_iterations']}
    max_run_wallclock:    47:30:00
    propagator:           executable
    gen_istates:          false
  data:
    west_data_file: west.h5
    datasets:
      - name:        pcoord
        scaleoffset: 4
      - name:        coord
        dtype:       float32
        scaleoffset: 3
    data_refs:
      segment:       $WEST_SIM_ROOT/traj_segs/{{segment.n_iter:06d}}/{{segment.seg_id:06d}}
      basis_state:   $WEST_SIM_ROOT/bstates/{{basis_state.auxref}}
      initial_state: $WEST_SIM_ROOT/istates/{{initial_state.iter_created}}/{{initial_state.state_id}}.rst
  plugins:
  executable:
    environ:
      PROPAGATION_DEBUG: 1
    datasets:
      - name:    coord
        enabled: false
    propagator:
      executable: $WEST_SIM_ROOT/westpa_scripts/runseg.sh
      stdout:     $WEST_SIM_ROOT/seg_logs/{{segment.n_iter:06d}}-{{segment.seg_id:06d}}.log
      stderr:     stdout
      stdin:      null
      cwd:        null
      environ:
        SEG_DEBUG: 1
    get_pcoord:
      executable: $WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh
      stdout:     /dev/null
      stderr:     stdout
    gen_istate:
      executable: $WEST_SIM_ROOT/westpa_scripts/gen_istate.sh
      stdout:     /dev/null
      stderr:     stdout
    post_iteration:
      enabled:    true
      executable: $WEST_SIM_ROOT/westpa_scripts/post_iter.sh
      stderr:     stdout
    pre_iteration:
      enabled:    false
      executable: $WEST_SIM_ROOT/westpa_scripts/pre_iter.sh
      stderr:     stdout
"""


# ─────────────────────────────────────────────────────────────────────────────
# ParMetaD — AMBER
# ─────────────────────────────────────────────────────────────────────────────

def gen_md_in_parmetad_amber(config):
    md = config["md"]
    pcoord_len = compute_pcoord_len(md["nsteps"], md["ntpr"])
    return f"""
 &cntrl

!!! INPUT
 irest = 1      ! restarting simulation
 ntx = 5        ! read in coordinates and velocities

!!! OUTPUT
 ntxo = 2       ! restart file format binary
 ntpr = {md['ntpr']}   ! write energy every {md['ntpr']} steps  (pcoord_len will be {pcoord_len})
 ntwx = {md['ntpr']}   ! write trajectory every {md['ntpr']} steps
 ntwr = {md['ntpr']}   ! write restart every {md['ntpr']} steps
 ioutfm = 1     ! binary NetCDF trajectories

!!! DYNAMICS
 nstlim = {md['nsteps']}  ! number of MD steps
 dt = {md['timestep']:.4f}   ! timestep in ps

!!! TEMPERATURE
 ntt = 3        ! Langevin thermostat
 temp0 = {md['temperature']:.1f}  ! target temperature (K)
 gamma_ln = 5   ! collision frequency 5/ps
 ig = RAND      ! random seed

!!! CONSTRAINTS
 ntc = 2        ! SHAKE on H-bonds
 ntf = 2        ! skip H-bond forces

!!! PLUMED (metadynamics)
 plumed = 1
 plumedfile = 'plumed.dat'

!!! NONBONDED (implicit solvent — adjust for explicit)
 cut = 9999.0
 rgbmax = 9999.0
 igb = 1        ! HCT GB implicit solvent

&end
"""


def gen_md_in_parmetad_amber_init(config):
    md = config["md"]
    return f"""
 &cntrl

!!! INPUT — initial run (no velocities)
 irest = 0
 ntx = 1

!!! OUTPUT
 ntxo = 2
 ntpr = {md['ntpr']}
 ntwx = {md['ntpr']}
 ntwr = {md['ntpr']}
 ioutfm = 1

!!! DYNAMICS
 nstlim = {md['nsteps']}
 dt = {md['timestep']:.4f}

!!! TEMPERATURE
 ntt = 3
 temp0 = {md['temperature']:.1f}
 gamma_ln = 5
 ig = RAND

!!! CONSTRAINTS
 ntc = 2
 ntf = 2

!!! PLUMED
 plumed = 1
 plumedfile = 'plumed.dat'

!!! NONBONDED
 cut = 9999.0
 rgbmax = 9999.0
 igb = 1

&end
"""


def gen_plumed_dat_metad(config):
    """Generate plumed.dat for metadynamics — AMBER or OpenMM backend"""
    pcoord = config["pcoord"]
    cvs = pcoord.get("cvs", [])
    md = config["md"]

    # Build CV definitions
    cv_defs = []
    print_args = []
    metad_args = []

    for i, cv in enumerate(cvs):
        cv_name = cv.get("name", f"cv{i+1}")
        cv_type = cv.get("type", "RMSD")
        cv_atoms = cv.get("atoms", "")
        sigma = cv.get("sigma", 0.05)
        height = cv.get("height", 1.2)

        if cv_type.upper() == "RMSD":
            cv_defs.append(f"{cv_name}: RMSD REFERENCE=reference.pdb TYPE=OPTIMAL ATOMS={cv_atoms if cv_atoms else '@CA'}")
        elif cv_type.upper() == "DISTANCE":
            cv_defs.append(f"{cv_name}: DISTANCE ATOMS={cv_atoms if cv_atoms else '1,10'}")
        elif cv_type.upper() == "TORSION":
            cv_defs.append(f"{cv_name}: TORSION ATOMS={cv_atoms if cv_atoms else '1,2,3,4'}")
        elif cv_type.upper() == "GYRATION":
            cv_defs.append(f"{cv_name}: GYRATION TYPE=RADIUS ATOMS={cv_atoms if cv_atoms else '@CA'}")
        else:
            cv_defs.append(f"# {cv_name}: {cv_type} — customize this CV definition")
            cv_defs.append(f"{cv_name}: DISTANCE ATOMS=1,2")

        print_args.append(cv_name)
        metad_args.append(f"ARG{i+1}={cv_name}")

    # First CV sigma/height (reuse or extend)
    sigma_list = ",".join(str(cv.get("sigma", 0.05)) for cv in cvs)
    height_val = cvs[0].get("height", 1.2) if cvs else 1.2
    pace_val = pcoord.get("hills_pace", 500)
    biasfactor = pcoord.get("biasfactor", 10)

    args_str = ",".join(cv.get("name", f"cv{i+1}") for i, cv in enumerate(cvs))
    metad_arg_str = f"ARG={args_str}"

    cv_block = "\n".join(cv_defs) if cv_defs else "# Define your CVs here\nrmsd: RMSD REFERENCE=reference.pdb TYPE=OPTIMAL"

    print_cv = args_str if args_str else "rmsd"
    sigma_str = sigma_list if sigma_list else "0.05"

    return f"""# PLUMED input for Well-Tempered Metadynamics
# Auto-generated by WE-Hybrid Setup Wizard
# Modify CV definitions and metadynamics parameters as needed

RESTART

# ── Collective Variables ──────────────────────────────────────────────────────
{cv_block}

# ── Well-Tempered Metadynamics ────────────────────────────────────────────────
metad: METAD \\
  {metad_arg_str if metad_arg_str.strip() else 'ARG=rmsd'} \\
  SIGMA={sigma_str} \\
  HEIGHT={height_val} \\
  PACE={pace_val} \\
  BIASFACTOR={biasfactor} \\
  TEMP={config['md']['temperature']:.1f} \\
  FILE=HILLS \\
  RESTART=YES

# ── Output ───────────────────────────────────────────────────────────────────
PRINT ARG={print_cv},metad.bias STRIDE=100 FILE=COLVAR
"""


def gen_plumed_init_dat_metad(config):
    """plumed_init.dat — first walker (no RESTART on METAD, fresh HILLS)"""
    base = gen_plumed_dat_metad(config)
    # Remove RESTART from METAD block for init
    return base.replace("  RESTART=YES\n", "")


def gen_runseg_sh_amber_parmetad(config):
    """runseg.sh for ParMetaD + AMBER"""
    westpa = config["westpa"]
    topology = config.get("system", {}).get("topology", "system.prmtop")

    if westpa["pcoord_ndim"] == 1:
        pcoord_paste = "tail -n +2 COLVAR | awk '{print $2}' > $WEST_PCOORD_RETURN"
    else:
        pcoord_paste = """paste <(tail -n +2 COLVAR | awk '{print $2}') \\
      <(tail -n +2 COLVAR | awk '{print $3}') > $WEST_PCOORD_RETURN"""

    return f"""#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

# Link common files
ln -sv $WEST_SIM_ROOT/common_files/{topology} .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  cp $WEST_SIM_ROOT/common_files/plumed.dat plumed.dat
  ln -sv $WEST_PARENT_DATA_REF/seg.rst ./parent.rst
  cp $WEST_PARENT_DATA_REF/HILLS HILLS 2>/dev/null || touch HILLS
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md_init.in > md.in
  cp $WEST_SIM_ROOT/common_files/plumed_init.dat plumed.dat
  ln -sv $WEST_PARENT_DATA_REF ./parent.rst
  touch HILLS
fi

# GPU assignment
export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${{CUDA_DEVICES[$WM_PROCESS_INDEX]}}

echo "RUNSEG: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# Run AMBER + PLUMED — retry loop for robustness
while ! grep -q "Final Performance Info" seg.log 2>/dev/null; do
    $PMEMD -O \\
      -i md.in \\
      -p {topology} \\
      -c parent.rst \\
      -r seg.rst \\
      -x seg.nc \\
      -o seg.log \\
      -inf seg.nfo \\
      -plumed plumed.dat
done

# Write progress coordinate
{pcoord_paste}

# Clean up temporaries
rm -f md.in seg.nfo
"""


def gen_get_pcoord_sh_amber_parmetad(config):
    westpa = config["westpa"]
    if westpa["pcoord_ndim"] == 1:
        pcoord_line = "tail -n +2 COLVAR | awk '{print $2}' > $WEST_PCOORD_RETURN"
    else:
        pcoord_line = "paste <(tail -n +2 COLVAR | awk '{print $2}') <(tail -n +2 COLVAR | awk '{print $3}') > $WEST_PCOORD_RETURN"

    return f"""#!/bin/bash
# get_pcoord.sh — extract progress coordinate from basis state

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

{pcoord_line}

if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi
"""


# ─────────────────────────────────────────────────────────────────────────────
# ParGaMD — AMBER
# ─────────────────────────────────────────────────────────────────────────────

def gen_md_in_parGaMD(config):
    """GaMD production md.in"""
    md = config["md"]
    gamd = config.get("gamd", {})
    pcoord_len = compute_pcoord_len(md["nsteps"], md["ntpr"])
    return f"""
 &cntrl

!!! INPUT
 irest = 1
 ntx = 5

!!! OUTPUT
 ntxo = 2
 ntpr = {md['ntpr']}   ! pcoord_len will be {pcoord_len}
 ntwx = {md['ntpr']}
 ntwr = {md['ntpr']}
 ioutfm = 1

!!! DYNAMICS
 nstlim = {md['nsteps']}
 dt = {md['timestep']:.4f}

!!! TEMPERATURE
 ntt = 3
 temp0 = {md['temperature']:.1f}
 gamma_ln = 5
 ig = RAND

!!! CONSTRAINTS
 ntc = 2
 ntf = 2

!!! NONBONDED
 cut = 9999.0
 rgbmax = 9999.0
 igb = 1

 !!! GaMD SETTINGS — uses parameters from gamd-restart.dat
 igamd = 3
 iE = {gamd.get('iE', 2)}
 irest_gamd = 1       ! restart GaMD (reads gamd-restart.dat)
 ntcmd = 0
 nteb = {md['nsteps']}
 ntave = {md['nsteps']}
 ntcmdprep = 0
 ntebprep = 0
 sigma0D = {gamd.get('sigma0D', 6.0)}
 sigma0P = {gamd.get('sigma0P', 6.0)}

&end
"""


def gen_md_init_in_parGaMD(config):
    """GaMD init md.in (new trajectory, no restart)"""
    md = config["md"]
    gamd = config.get("gamd", {})
    return f"""
 &cntrl

!!! INPUT — initial walker
 irest = 0
 ntx = 1

!!! OUTPUT
 ntxo = 2
 ntpr = {md['ntpr']}
 ntwx = {md['ntpr']}
 ntwr = {md['ntpr']}
 ioutfm = 1

!!! DYNAMICS
 nstlim = {md['nsteps']}
 dt = {md['timestep']:.4f}

!!! TEMPERATURE
 ntt = 3
 temp0 = {md['temperature']:.1f}
 gamma_ln = 5
 ig = RAND

!!! CONSTRAINTS
 ntc = 2
 ntf = 2

!!! NONBONDED
 cut = 9999.0
 rgbmax = 9999.0
 igb = 1

 !!! GaMD SETTINGS
 igamd = 3
 iE = {gamd.get('iE', 2)}
 irest_gamd = 1
 ntcmd = 0
 nteb = {md['nsteps']}
 ntave = {md['nsteps']}
 ntcmdprep = 0
 ntebprep = 0
 sigma0D = {gamd.get('sigma0D', 6.0)}
 sigma0P = {gamd.get('sigma0P', 6.0)}

&end
"""


def gen_cmd_md_in(config):
    """cGaMD preliminary run input — generates gamd-restart.dat"""
    md = config["md"]
    gamd = config.get("gamd", {})
    nstlim_cmd = md.get("cmd_nsteps", 2000000)
    return f"""
 &cntrl
 irest = 0,
 ntx = 1,
 ntxo = 2,
 ntpr = 1000,
 ntwx = 1000,
 ntwr = 1000,
 ioutfm = 1,
 nstlim = {nstlim_cmd},
 dt = {md['timestep']:.4f},
 ntt = 3,
 temp0 = {md['temperature']:.1f},
 gamma_ln = 5,
 ig = -1,
 ntc = 2,
 ntf = 2,
 cut = 9999.0,
 rgbmax = 9999.0,
 igb = 1,

 !!! GaMD — equilibration to get boost parameters
 igamd = 3,
 iE = {gamd.get('iE', 2)},
 irest_gamd = 0,
 ntcmd = 1000000,
 nteb = {nstlim_cmd},
 ntave = 50000,
 ntcmdprep = 200000,
 ntebprep = 200000,
 sigma0D = {gamd.get('sigma0D', 6.0)},
 sigma0P = {gamd.get('sigma0P', 6.0)},
&end
"""


def gen_run_cmd_sh(config):
    """SLURM script for cGaMD pre-run"""
    hpc = config["hpc"]
    email_line = f"#SBATCH --mail-user={hpc['email']}" if hpc.get('email') else "#SBATCH --mail-user=your@email.com"
    return f"""#!/bin/bash
#SBATCH --job-name="cGaMD_prerun"
#SBATCH --output="job_cmd.out"
#SBATCH --error="job_cmd.err"
#SBATCH --partition={hpc.get('cmd_partition', 'gpu-shared')}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem={hpc.get('mem', '50G')}
#SBATCH --account={hpc['account']}
#SBATCH --no-requeue
{email_line}
#SBATCH --mail-type=ALL
#SBATCH -t {hpc.get('cmd_walltime', '48:00:00')}

module purge
module load shared
module load gpu/0.15.4
module load slurm
module load openmpi/4.0.4
module load cuda/{hpc.get('cuda_version', '11.0.2')}
module load amber/{hpc.get('amber_version', '20-patch15')}

export PATH=$PATH:$HOME/bin
source $AMBERHOME/amber.sh

# Run conventional GaMD — produces gamd-restart.dat and md_cmd.rst
pmemd.cuda -O -i md_cmd.in -o md_cmd.out -p chignolin.prmtop \\
           -c chignolin.rst -r md_cmd.rst -x md_cmd.nc -gamd gamd.log

# Copy gamd-restart.dat to common_files for WE run
cp gamd.log ../common_files/gamd-restart.dat 2>/dev/null || true
echo "cGaMD finished — check gamd.log and md_cmd.rst"
echo "Next: submit WE run with sbatch --dependency=afterok:\\$SLURM_JOBID ../run_WE.sh"
"""


def gen_runseg_sh_parGaMD(config):
    """runseg.sh for ParGaMD"""
    westpa = config["westpa"]
    if westpa["pcoord_ndim"] == 1:
        pcoord_line = "tail -n +2 rmsd.dat | awk '{print $2}' > $WEST_PCOORD_RETURN"
    else:
        pcoord_line = """paste <(tail -n +2 rmsd.dat | awk '{print $2}') \\
      <(tail -n +2 rg.dat | awk '{print $2}') > $WEST_PCOORD_RETURN"""

    return f"""#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

ln -sv $WEST_SIM_ROOT/common_files/chignolin.prmtop .
ln -sv $WEST_SIM_ROOT/common_files/gamd-restart.dat .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  ln -sv $WEST_PARENT_DATA_REF/seg.rst ./parent.rst
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md_init.in > md.in
  ln -sv $WEST_PARENT_DATA_REF ./parent.rst
fi

# GPU assignment
export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${{CUDA_DEVICES[$WM_PROCESS_INDEX]}}
echo "RUNSEG: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# Run GaMD with restart loop
while ! grep -q "Final Performance Info" seg.log 2>/dev/null; do
    $PMEMD -O \\
      -i md.in \\
      -p chignolin.prmtop \\
      -c parent.rst \\
      -r seg.rst \\
      -x seg.nc \\
      -o seg.log \\
      -inf seg.nfo \\
      -gamd gamd.log
done

# Extract progress coordinates using cpptraj
COMMAND="parm chignolin.prmtop\\n"
COMMAND="${{COMMAND}}trajin $WEST_CURRENT_SEG_DATA_REF/parent.rst\\n"
COMMAND="${{COMMAND}}trajin $WEST_CURRENT_SEG_DATA_REF/seg.nc\\n"
COMMAND="${{COMMAND}}reference $WEST_SIM_ROOT/common_files/chignolin.pdb\\n"
COMMAND="${{COMMAND}}rms ca-rmsd @CA reference out rmsd.dat mass\\n"
COMMAND="${{COMMAND}}radgyr ca-rg @CA out rg.dat mass\\n"
COMMAND="${{COMMAND}}go\\n"

echo -e $COMMAND | $CPPTRAJ

{pcoord_line}

rm -f md.in seg.nfo
"""


def gen_get_pcoord_sh_parGaMD(config):
    return """#!/bin/bash
# get_pcoord.sh — extract pcoord from basis state for ParGaMD

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

# For basis states, return a placeholder pcoord
# This is called before any dynamics; actual pcoord computed in runseg.sh
echo "0.0 0.0" > $WEST_PCOORD_RETURN
"""


def gen_reweight_sh(config):
    westpa = config["westpa"]
    bins = westpa["bin_boundaries"]
    ndim = westpa["pcoord_ndim"]
    if ndim == 2:
        b1 = bins[0]
        b2 = bins[1]
        max1 = [v for v in b1 if isinstance(v, (int, float))][-1]
        max2 = [v for v in b2 if isinstance(v, (int, float))][-1]
        spacing1 = round(max1 / 50, 2)
        spacing2 = round(max2 / 50, 2)
        return f"""#!/bin/bash
# reweight-2d.sh — reweight ParGaMD trajectory to get free energy surface
# Usage: ./reweight-2d.sh <cutoff1> <cutoff2> <spacing1> <spacing2> <output.dat> <temperature>

CUTOFF1=${{1:-{max1}}}
CUTOFF2=${{2:-{max2}}}
SPACING1=${{3:-{spacing1}}}
SPACING2=${{4:-{spacing2}}}
OUTFILE=${{5:-output.dat}}
TEMP=${{6:-{config['md']['temperature']}}}

echo "Running 2D reweighting..."
python PyReweighting-2D.py \\
    -input $OUTFILE \\
    -Xmax $CUTOFF1 -Ymax $CUTOFF2 \\
    -discX $SPACING1 -discY $SPACING2 \\
    -T $TEMP \\
    -Emax 20 \\
    -job reweight_ME \\
    -weight weights.dat

echo "Done — check pmf-c2-ME.dat for the free energy surface"
"""
    else:
        return """#!/bin/bash
# reweight-1d.sh — reweight ParGaMD trajectory
# Usage: ./reweight-1d.sh <cutoff> <spacing> <output.dat> <temperature>
CUTOFF=${1:-10}
SPACING=${2:-0.1}
OUTFILE=${3:-output.dat}
TEMP=${4:-300}

python PyReweighting-2D.py -input $OUTFILE -Xmax $CUTOFF -discX $SPACING -T $TEMP -Emax 20 -job reweight_ME -weight weights.dat
"""


# ─────────────────────────────────────────────────────────────────────────────
# OpenMM files
# ─────────────────────────────────────────────────────────────────────────────

def gen_env_sh_openmm(config):
    hpc = config["hpc"]
    paths = config["paths"]
    plumed_line = f"export PLUMED_KERNEL={paths['plumed_kernel_dir']}/libplumedKernel.so" if paths.get("plumed_kernel_dir") else "# export PLUMED_KERNEL=/path/to/libplumedKernel.so"

    return f"""#!/bin/bash

source ~/.bash_profile
module purge
module load shared
module load gpu/0.15.4
module load slurm
module load openmpi/4.0.4
module load cuda/{hpc.get('cuda_version', '11.0.2')}
conda activate {hpc['conda_env']}

export PATH=$PATH:$HOME/bin
export PYTHONPATH=$(which python)

if [[ -z "$WEST_SIM_ROOT" ]]; then
    export WEST_SIM_ROOT="$PWD"
fi

export SIM_NAME=$(basename $WEST_SIM_ROOT)
echo "simulation $SIM_NAME root is $WEST_SIM_ROOT"

{plumed_line}

export NODELOC={hpc['scratch_dir']}
export USE_LOCAL_SCRATCH=1

export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT_FACTOR=300

export BASH=$SWROOT/bin/bash
export LN=$SWROOT/bin/ln
export CP=$SWROOT/bin/cp
export RM=$SWROOT/bin/rm
export SED=$SWROOT/bin/sed
export CAT=$SWROOT/bin/cat
export TAR=$SWROOT/bin/tar
export AWK=$SWROOT/usr/bin/awk
export PASTE=$SWROOT/usr/bin/paste
export MKDIR=$SWROOT/bin/mkdir
"""


def gen_metadynamics_config_yaml(config):
    md = config["md"]
    pcoord = config["pcoord"]
    cv_names = [cv.get("name", f"cv{i+1}") for i, cv in enumerate(pcoord.get("cvs", []))]
    plumed_file = "plumed.dat"

    return f"""# Metadynamics Simulation Configuration — auto-generated by WE-Hybrid Setup Wizard

mode: "restart"  # "init" for fresh start, "restart" to continue

input_files:
  prmtop: "chignolin.prmtop"
  plumed: "{plumed_file}"
  checkpoint: "final_checkpoint.chk"   # for restart mode
  # inpcrd: "chignolin_box.inpcrd"     # uncomment for init mode

md_parameters:
  nsteps: {md['nsteps']}
  output_freq: {md['ntpr']}
  temperature: {md['temperature']:.1f}
  timestep: {md['timestep']:.4f}
  platform: "CUDA"
  implicit_solvent: "HCT"
  nonbonded_method: "NoCutoff"
  constraints: "HBonds"
  hydrogen_mass: 1.5

output_files:
  dcd_trajectory: "trajectory.dcd"
  pdb_snapshots: "final.pdb"
  final_pdb_only: true
  output_checkpoint: "final_checkpoint.chk"
  nc_restart: "final_restart.nc"
  log_file: "metadynamics.log"

advanced:
  minimize_energy: false
  checkpoint_freq: 10000
  backup_checkpoints: true
"""


def gen_run_openmm_sh(config):
    hpc = config["hpc"]
    return f"""#!/bin/bash
# run_openmm.sh — run a single OpenMM+PLUMED metadynamics segment

source $WEST_SIM_ROOT/env.sh

export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${{CUDA_DEVICES[$WM_PROCESS_INDEX]}}

python chignolin_metadynamics_yaml.py --config metadynamics_config.yaml
"""


def gen_runseg_sh_openmm(config):
    westpa = config["westpa"]
    if westpa["pcoord_ndim"] == 1:
        pcoord_line = "tail -n +2 COLVAR | awk '{print $2}' > $WEST_PCOORD_RETURN"
    else:
        pcoord_line = "paste <(tail -n +2 COLVAR | awk '{print $2}') <(tail -n +2 COLVAR | awk '{print $3}') > $WEST_PCOORD_RETURN"

    return f"""#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

# Link common files
ln -sv $WEST_SIM_ROOT/common_files/chignolin.prmtop .
ln -sv $WEST_SIM_ROOT/common_files/chignolin_box.inpcrd .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  cp $WEST_SIM_ROOT/common_files/metadynamics_config.yaml .
  cp $WEST_SIM_ROOT/common_files/plumed.dat .
  ln -sv $WEST_PARENT_DATA_REF/final_checkpoint.chk ./final_checkpoint.chk
  cp $WEST_PARENT_DATA_REF/HILLS HILLS 2>/dev/null || touch HILLS
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  cp $WEST_SIM_ROOT/common_files/config_init.yaml metadynamics_config.yaml
  cp $WEST_SIM_ROOT/common_files/plumed_init.dat plumed.dat
  touch HILLS
fi

ln -sv $WEST_SIM_ROOT/common_files/chignolin_metadynamics_yaml.py .

# GPU assignment
export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${{CUDA_DEVICES[$WM_PROCESS_INDEX]}}
echo "RUNSEG: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# Run OpenMM
while ! [ -f final_checkpoint.chk ]; do
    python chignolin_metadynamics_yaml.py --config metadynamics_config.yaml
done

{pcoord_line}
"""


def gen_run_we_sh_openmm(config):
    hpc = config["hpc"]
    email_line = f"#SBATCH --mail-user={hpc['email']}" if hpc.get('email') else "#SBATCH --mail-user=your@email.com"

    return f"""#!/bin/bash
#SBATCH --job-name="ParMetaD_OpenMM"
#SBATCH --output="job.out"
#SBATCH --error="job.err"
#SBATCH --partition={hpc['partition']}
#SBATCH --nodes={hpc['nodes']}
#SBATCH --gpus={int(hpc['nodes']) * int(hpc['gpus_per_node'])}
#SBATCH --ntasks-per-node=1
#SBATCH --mem={hpc.get('mem', '50G')}
#SBATCH --account={hpc['account']}
#SBATCH --no-requeue
{email_line}
#SBATCH --mail-type=ALL
#SBATCH -t {hpc['walltime']}

set -x
cd $SLURM_SUBMIT_DIR
source ~/.bashrc
module purge
module load shared
module load gpu/0.15.4
module load slurm
module load openmpi/4.0.4
module load cuda/{hpc.get('cuda_version', '11.0.2')}
conda activate {hpc['conda_env']}

export PLUMED_KERNEL={config['paths'].get('plumed_kernel_dir', '/path/to/plumed/lib')}/libplumedKernel.so
export WEST_SIM_ROOT=$SLURM_SUBMIT_DIR
cd $WEST_SIM_ROOT

./init.sh
echo "init.sh ran"
source env.sh || exit 1

SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info.json

num_gpu_per_node={hpc['gpus_per_node']}
rm -rf nodefilelist.txt
scontrol show hostname $SLURM_JOB_NODELIST > nodefilelist.txt

w_run --work-manager=zmq --n-workers=0 \\
      --zmq-mode=master \\
      --zmq-write-host-info=$SERVER_INFO \\
      --zmq-comm-mode=tcp &> west-$SLURM_JOBID-local.log &

for ((n=0; n<60; n++)); do
    if [ -e $SERVER_INFO ] ; then break; fi
    sleep 1
done

if ! [ -e $SERVER_INFO ] ; then
    echo 'ZMQ master failed to start'
    exit 1
fi

CUDA_LIST=$(seq -s, 0 $((num_gpu_per_node - 1)))
export CUDA_VISIBLE_DEVICES=$CUDA_LIST

for node in $(cat nodefilelist.txt); do
    ssh -o StrictHostKeyChecking=no $node \\
        $PWD/node.sh $SLURM_SUBMIT_DIR $SLURM_JOBID $node $CUDA_VISIBLE_DEVICES \\
        --work-manager=zmq --n-workers=$num_gpu_per_node \\
        --zmq-mode=client \\
        --zmq-read-host-info=$SERVER_INFO \\
        --zmq-comm-mode=tcp &
done
wait
"""


# ─────────────────────────────────────────────────────────────────────────────
# MASTER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_files(config):
    """
    Generate all simulation files based on config dict.
    Returns dict of {relative_path: file_content_string}
    """
    method = config["method"]   # "parmetad" or "parGaMD"
    backend = config["backend"] # "amber" or "openmm"

    files = {}

    # ── Shared files ──────────────────────────────────────────────────────────
    files["bstates/bstates.txt"] = gen_bstates_txt(config)
    files["bstates/empty.txt"] = "# Place your basis state restart file(s) here\n"
    files["init.sh"] = gen_init_sh(config)
    files["node.sh"] = gen_node_sh(config)
    files["west.cfg"] = gen_west_cfg(config)
    files["westpa_scripts/post_iter.sh"] = gen_post_iter_sh()
    files["westpa_scripts/gen_istate.sh"] = gen_gen_istate_sh()
    files["westpa_scripts/tar_segs.sh"] = gen_tar_segs_sh()
    files["westpa_scripts/cat_trajectory.py"] = gen_cat_trajectory_py()
    files["westpa_scripts/empty.txt"] = "# westpa_scripts directory\n"
    files["common_files/empty.txt"] = "# Place topology and structure files here\n"

    # ── README for post-generation steps ─────────────────────────────────────
    files["SETUP_INSTRUCTIONS.md"] = gen_setup_instructions(config)

    # ── Method + Backend specific ─────────────────────────────────────────────
    if method == "parmetad" and backend == "amber":
        files["env.sh"] = gen_env_sh_amber(config)
        files["run_WE.sh"] = gen_run_we_sh_amber(config)
        files["common_files/md.in"] = gen_md_in_parmetad_amber(config)
        files["common_files/md_init.in"] = gen_md_in_parmetad_amber_init(config)
        files["common_files/plumed.dat"] = gen_plumed_dat_metad(config)
        files["common_files/plumed_init.dat"] = gen_plumed_init_dat_metad(config)
        files["westpa_scripts/runseg.sh"] = gen_runseg_sh_amber_parmetad(config)
        files["westpa_scripts/get_pcoord.sh"] = gen_get_pcoord_sh_amber_parmetad(config)

    elif method == "parmetad" and backend == "openmm":
        files["env.sh"] = gen_env_sh_openmm(config)
        files["run_WE.sh"] = gen_run_we_sh_openmm(config)
        files["common_files/metadynamics_config.yaml"] = gen_metadynamics_config_yaml(config)
        files["common_files/config_init.yaml"] = gen_metadynamics_config_yaml(config).replace(
            'mode: "restart"', 'mode: "init"'
        )
        files["common_files/plumed.dat"] = gen_plumed_dat_metad(config)
        files["common_files/plumed_init.dat"] = gen_plumed_init_dat_metad(config)
        files["common_files/run_openmm.sh"] = gen_run_openmm_sh(config)
        files["westpa_scripts/runseg.sh"] = gen_runseg_sh_openmm(config)
        files["westpa_scripts/get_pcoord.sh"] = gen_get_pcoord_sh_amber_parmetad(config)

    elif method == "parGaMD":
        files["env.sh"] = gen_env_sh_amber(config)
        files["run_WE.sh"] = gen_run_we_sh_amber(config)
        files["common_files/md.in"] = gen_md_in_parGaMD(config)
        files["common_files/md_init.in"] = gen_md_init_in_parGaMD(config)
        files["cMD/md_cmd.in"] = gen_cmd_md_in(config)
        files["cMD/run_cmd.sh"] = gen_run_cmd_sh(config)
        files["cMD/empty.txt"] = "# Place chignolin.prmtop and chignolin.rst here\n"
        files["westpa_scripts/runseg.sh"] = gen_runseg_sh_parGaMD(config)
        files["westpa_scripts/get_pcoord.sh"] = gen_get_pcoord_sh_parGaMD(config)
        files["reweight-2d.sh"] = gen_reweight_sh(config)

    return files


def generate_zip(config):
    """Return a BytesIO zip buffer containing all generated files."""
    files = generate_files(config)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    buf.seek(0)
    return buf


def gen_setup_instructions(config):
    method = config["method"]
    backend = config["backend"]
    hpc = config["hpc"]
    md = config["md"]
    pcoord_len = compute_pcoord_len(md["nsteps"], md["ntpr"])

    reweight_line = "# Then reweight:\nbash reweight-2d.sh" if method == "parGaMD" else ""
    cmd_files_line = "- `cMD/chignolin.prmtop`, `cMD/chignolin.rst` — for cGaMD pre-run" if method == "parGaMD" else ""
    chmod_cmd_extra = "cMD/run_cmd.sh" if method == "parGaMD" else ""

    pre_steps = ""
    if method == "parGaMD":
        pre_steps = f"""
## Step 1: Run the cGaMD pre-run (REQUIRED for ParGaMD)

```bash
cd cMD/
# Copy your topology and starting structure here:
#   chignolin.prmtop, chignolin.rst

sbatch run_cmd.sh
# Note the job ID from: squeue -u $USER
JOB_ID=<YOUR_JOB_ID>
```
"""
        we_step = "## Step 2: Submit the WE simulation as a dependency"
        we_cmd = f"sbatch --dependency=afterok:$JOB_ID run_WE.sh"
    else:
        pre_steps = ""
        we_step = "## Step 1: Copy your system files and submit"
        we_cmd = "sbatch run_WE.sh"

    plumed_note = ""
    if method == "parmetad":
        plumed_note = """
> **PLUMED CVs**: Edit `common_files/plumed.dat` to define your collective variables.
> The HILLS file accumulates bias across all walkers via the WE framework.
"""

    return f"""# WE-Hybrid Simulation Setup Instructions
Auto-generated by WE-Hybrid Setup Wizard

## Configuration Summary
- Method: {method}
- Backend: {backend}
- Nodes: {hpc['nodes']} × {hpc['gpus_per_node']} GPUs = {int(hpc['nodes']) * int(hpc['gpus_per_node'])} total walkers
- nstlim={md['nsteps']}, ntpr={md['ntpr']} → pcoord_len={pcoord_len} ✓
- Temperature: {md['temperature']} K

## Files to place manually (NOT generated — system-specific)
- `common_files/chignolin.prmtop` — AMBER topology
- `common_files/chignolin.pdb` — Reference PDB (for RMSD)
- `bstates/bstate.rst` — Basis state restart file
{cmd_files_line}
{plumed_note}
{pre_steps}
{we_step}

```bash
# Transfer files to HPC
scp -r we_simulation/ {hpc['account']}@login.expanse.sdsc.edu:{hpc['scratch_dir']}/

# SSH to cluster and go to your simulation dir
ssh {hpc['account']}@login.expanse.sdsc.edu
cd {hpc['scratch_dir']}/we_simulation

# Make scripts executable
chmod +x *.sh westpa_scripts/*.sh {chmod_cmd_extra}

{we_cmd}
```

## Monitoring
```bash
# Check job status
squeue -u $USER

# Monitor WE progress
w_ipa -r west.h5 --force

# View log
tail -f west-*.log
```

## Post-processing (after simulation)
```bash
bash run_data.sh   # extract gamd.log and PC.dat
{reweight_line}
```

## Key constraint (always verified by wizard)
`pcoord_len` = `nstlim` / `ntpr` + 1 = {md['nsteps']} / {md['ntpr']} + 1 = **{pcoord_len}**
This value is set consistently in both `west.cfg` and `common_files/md.in`.
"""
