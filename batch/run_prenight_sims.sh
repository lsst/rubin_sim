#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=auxtel_prenight_daily   # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims.err  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=4G                       # Requested memory
#SBATCH --time=1:00:00                 # Wall time (hh:mm:ss)

echo "******** START of run_prenight_sims.sh **********"

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# SLAC S3DF - source all files under ~/.profile.d
if [[ -e ~/.profile.d && -n "$(ls -A ~/.profile.d/)" ]]; then
  source <(cat $(find -L  ~/.profile.d -name '*.conf'))
fi

date --iso=s

source /sdf/group/rubin/sw/w_latest/loadLSST.sh
conda activate /sdf/data/rubin/shared/scheduler/envs/prenight

set -o xtrace

export AWS_PROFILE=prenight
WORK_DIR=$(date '+/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%S' --utc)
echo "Working in $WORK_DIR"
mkdir ${WORK_DIR}
cd ${WORK_DIR}

# Get ts_ocs_config
TS_CONFIG_OCS_VERSION=$(obs_version_at_time ts_config_ocs)
curl --location --output ts_config_ocs.zip https://github.com/lsst-ts/ts_config_ocs/archive/${TS_CONFIG_OCS_VERSION}.zip
unzip ts_config_ocs.zip
mv $(find . -maxdepth 1 -type d -name ts_config_ocs\*) ts_config_ocs

# Install required python packages
PACKAGE_DIR=$(readlink -f ${WORK_DIR}/packages)
mkdir ${PACKAGE_DIR}

# Get the scheduler version from the EFD and install it.
RUBIN_SCHEDULER_TAG=v$(obs_version_at_time rubin_scheduler)
pip install --no-deps --target=${PACKAGE_DIR} git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_TAG}

# Cannot get ts_fbs_utils from the EFD, so just guess the highest semantic version tag in the repo.
TS_FBS_UTILS_TAG=$(curl -s https://api.github.com/repos/lsst-ts/ts_fbs_utils/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
pip install --no-deps --target=${PACKAGE_DIR} git+https://github.com/lsst-ts/ts_fbs_utils.git@${TS_FBS_UTILS_TAG}

# Get the scheduler configuration script
SCHEDULER_CONFIG_SCRIPT=$(scheduler_config_at_time latiss)

# Get the path to prenight_sim as provided by the current environment,
# so we do not accidentally run one from the adjusted PATH below.
PRENIGHT_SIM=$(which prenight_sim)

export PYTHONPATH=${PACKAGE_DIR}:${PYTHONPATH}
export PATH=${PACKAGE_DIR}/bin:${PATH}
printenv > env.out
date --iso=s
time ${PRENIGHT_SIM} --scheduler auxtel.pickle.xz --opsim None --script ${SCHEDULER_CONFIG_SCRIPT}
date --iso=s
echo "******* END of run_prenight_sims.sh *********"
