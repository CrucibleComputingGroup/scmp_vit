#!/usr/bin/env bash
# Run a vit_sc command on a GPU node with the right cwd/env, no matter where
# ssh drops us. Usage:  ssh <node> 'bash /abs/path/cls/_run.sh <cmd...>'
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
export IMAGENET_ROOT=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
exec "$@" 2>&1
