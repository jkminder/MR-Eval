set -o pipefail;

DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build -t harmbench:latest -f dockerfile-harmbench . 2>&1 | tee build-harmbench.log
