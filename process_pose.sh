#!/bin/bash
BASEDIR=$(dirname "$0")

# activate the conda environment
CONDA_DIR="$(conda info --base)"
. "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate contactopt

echo $CONDA_PREFIX

echo cd $BASEDIR
cd $BASEDIR

# set python path so contactopt module is found
export PYTHONPATH=$BASEDIR

echo python contactopt/run_clean_GRAB.py $@
python contactopt/run_clean_GRAB.py $@
