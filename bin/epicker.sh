#!/bin/bash


SOURCE="${BASH_SOURCE[0]}"
while [ -h "${SOURCE}" ]; do
  SCRIPTDIR="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"
  SOURCE="$(readlink "${SOURCE}")"
  [[ ${SOURCE} != /* ]] && SOURCE="${SCRIPTDIR}/${SOURCE}"
done
PROOT="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"


if (( $#<=0 )); then
	echo "EPicker : CenterNet-Based Particle Picking for CryoEM"
        echo "Usage: "
	echo "  EPicker [options]"
	echo "List of options:"
	echo " --data         Image list(.thi) or folder contains images to be picked."
	echo " --load_model   Trained model for picking."
	echo " --output       Folder of output coordinate files, Optional, default the current folder."
	echo " --output_type  Format of coordinate file: thi/star/box/coord . Optional, default .thi."	
	echo " --mode         Picking mode: particle/vesicle/fiber. Optional, default particle."
	echo " --K            Maximum number of paticles in a micrograph. Optional, default 2000."
	echo " --vis_thresh   Threshold of picking score: 0~1. Optional, default 0"
	echo " --min_distance The minimal distance of adjacent partices, default 0"
	echo " --visual       Output diagnostic images."
	echo " --edge         Width of edge after rescale to 1024 pixels, optional, default 25."	
	echo " --gpus         GPU device ID, multiple GPUs is not supported now. Optional, default 0."
	echo " --ang_thresh   Maximum curvature for tracing fiber. Only for fiber mode, default 0.3"  

	echo " "

	exit 0
fi

PYTHON=${PROOT}/../../python/bin/python3
env -i $PYTHON ${PROOT}/../pick.py $*
