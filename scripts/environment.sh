#! /bin/bash -l

# User specified paths for python environment and NEB program path
ENV_PATH="/home/lmeeder/environments/neb-env/bin/activate"
NEB_PATH="/home/lmeeder/Programme/goeneb"

if [ -z "$TMP_DIR" ]; then
    TMP_DIR="/scr/${USER}"
fi

# Temp directory
JOB_ID="${SLURM_JOB_ID:-$$}"
NEB_TMPDIR=${TMP_DIR}/${JOB_ID}
mkdir -p ${NEB_TMPDIR}
export NEB_TMPDIR

# Executables
if [ -z "$GAUSS_EXE" ]; then
    export GAUSS_EXE="$(which g16)"
fi
if [ -z "$MOL_EXE" ]; then
    export MOL_EXE="$(which molpro)"
fi
if [ -z "$ORCA_EXE" ]; then
    export ORCA_EXE="$(which orca)"
fi
if [ -z "$MPI_PATH" ]; then
    export MPI_PATH="$(which mpirun)"
fi
export NEB_PATH

export OMP_STACKSIZE=4G

# Python environment
source ${ENV_PATH}