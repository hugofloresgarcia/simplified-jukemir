source env.sh

pushd ..
set -e
HOST_CACHE=$(python -c "from jukemir import CACHE_DIR; print(CACHE_DIR)")
echo $HOST_CACHE
popd

DOCKER_CPUS=$(python3 -c "import os; cpus=os.sched_getaffinity(0); print(','.join(map(str,cpus)))")
DOCKER_GPUS=$(nvidia-smi -L | python3 -c "import sys; print(','.join([l.strip().split()[-1][:-1] for l in list(sys.stdin)]))")
DOCKER_CPU_ARG="--cpuset-cpus ${DOCKER_CPUS}"
DOCKER_GPU_ARG="--gpus device=${DOCKER_GPUS}"

docker run \
  -it \
  --rm \
  -d \
  ${DOCKER_CPU_ARG} \
  ${DOCKER_GPU_ARG} \
  --name ${DOCKER_NAME} \
  -u $(id -u):$(id -g) \
  -v $HOST_CACHE:/jukemir/cache \
  -v $(pwd)/../jukemir:/jukemir/jukemir \
  -v $(pwd)/../notebooks:/jukemir/notebooks \
  -v $(pwd)/../scripts:/jukemir/scripts \
  -v $(pwd)/../tests:/jukemir/tests \
  -v ~/.local:/.local \
  -p 8888:8888 \
  ${DOCKER_NAMESPACE}/${DOCKER_TAG} \
  bash
