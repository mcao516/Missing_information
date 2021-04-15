#!/bin/sh
source ~/env37/bin/activate

CUSTOM_PORT=7991
ssh -f -N -R $CUSTOM_PORT:localhost:$CUSTOM_PORT beluga3
export JUPYTER_RUNTIME_DIR=$SLURM_TMPDIR
jupyter lab --port=$CUSTOM_PORT --no-browser
