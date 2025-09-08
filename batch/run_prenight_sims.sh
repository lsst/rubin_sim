#!/usr/bin/env bash
#SBATCH --account=rubin:developers      # Account name
#SBATCH --job-name=auxtel_prenight_daily   # Job name
#SBATCH --output=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out # Output file (stdout)
#SBATCH --error=/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out  # Error file (stderr)
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=8G                       # Requested memory
#SBATCH --time=1:30:00                 # Wall time (hh:mm:ss)

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

LATEST_TAGGED_STACK=$(find /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib -maxdepth 1 -regex .*/'v[0-9]+\.[0-9]+\.[0-9]+' -printf "%f\n" | sort -V | tail -1)
echo "Using stack ${LATEST_TAGGED_STACK}"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/${LATEST_TAGGED_STACK}/loadLSST-ext.sh

set -o xtrace

export AWS_PROFILE=prenight
WORK_DIR=$(date '+/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%S' --utc)
echo "Working in $WORK_DIR"
mkdir ${WORK_DIR}
cd ${WORK_DIR}

# Install required python packages
PACKAGE_DIR=$(readlink -f ${WORK_DIR}/packages)
mkdir ${PACKAGE_DIR}
export PYTHONPATH=${PACKAGE_DIR}:${PYTHONPATH}
export PATH=${PACKAGE_DIR}/bin:${PATH}

# Install most of the packages we need, carefully controlling the
# versions and not installing dependincies, such that versions not
# explicitly installed or specified come from either the base LSST
# environment or the pip install of the tagged rubin_scheduler below.

# Cannot get ts_fbs_utils from the EFD, so just guess the highest semantic version tag in the repo.
# A "reference" can be a tag, hash, or branch.
TS_FBS_UTILS_REFERENCE=$(curl -s https://api.github.com/repos/lsst-ts/ts_fbs_utils/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
RUBIN_SIM_REFERENCE="v2.3.0"
SCHEDVIEW_REFERENCE="v0.19.0"
SIMS_SV_SURVEY_REFERENCE="v0.1.1"
RUBIN_NIGHTS_REFERENCE="v0.4.0"

pip install --no-deps --target=${PACKAGE_DIR} \
  git+https://github.com/lsst/rubin_sim.git@${RUBIN_SIM_REFERENCE} \
  git+https://github.com/lsst/schedview.git@${SCHEDVIEW_REFERENCE} \
  git+https://github.com/lsst-ts/ts_fbs_utils.git@${TS_FBS_UTILS_REFERENCE} \
  git+https://github.com/lsst-sims/sims_sv_survey.git@${SIMS_SV_SURVEY_REFERENCE} \
  git+https://github.com/lsst-sims/rubin_nights.git@${RUBIN_NIGHTS_REFERENCE} \
  lsst-resources

# Get the scheduler version from the EFD and install it.
# We have to do this after the others, because we want
# the version of obs_version_at_time supplied by the
# version of schedview we specify.
# Go ahead and install any missing dependencies as well.
RUBIN_SCHEDULER_REFERENCE=v$(obs_version_at_time rubin_scheduler)
echo "Using rubin_scheduler $RUBIN_SCHEDULER_REFERENCE"
pip install --ignore-installed --no-deps --target=${PACKAGE_DIR} git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_REFERENCE}

# Get the scheduler configuration script
# It lives in ts_ocs_config
TS_CONFIG_OCS_REPO="https://github.com/lsst-ts/ts_config_ocs"
TS_CONFIG_OCS_REFERENCE=$(obs_version_at_time ts_config_ocs)
echo "Using ts_config_ocs ${TS_CONFIG_OCS_REFERENCE}"
curl --location --output ts_config_ocs.zip ${TS_CONFIG_OCS_REPO}/archive/${TS_CONFIG_OCS_REFERENCE}.zip
unzip ts_config_ocs.zip
mv $(find . -maxdepth 1 -type d -name ts_config_ocs\*) ts_config_ocs

SCHEDULER_CONFIG_SCRIPT="ts_config_ocs/Scheduler/feature_scheduler/maintel/fbs_config_sv_survey.py"

export SIM_ARCHIVE_LOG_FILE=${WORK_DIR}/sim_archive_log.txt
export PRENIGHT_LOG_FILE=${WORK_DIR}/prenight_log.txt
export DAYOBS="$(date -u +'%Y%m%d')"
export ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/"

echo "Fetching completed visits"
date --iso=s
fetch_sv_visits ${DAYOBS} completed_visits.db ~/.usdf_access_token

echo "Creating scheduler pickle"
date --iso=s
make_sv_scheduler scheduler.p --opsim completed_visits.db --config-script ${SCHEDULER_CONFIG_SCRIPT}

echo "Creating model observatory"
date --iso=s
make_model_observatory observatory.p

echo "Creating band scheduler"
date --iso=s
make_band_scheduler band_scheduler.p

echo "Running nominal SV simulation"
OPSIMRUN="prenight_nominal_$(date --iso=s)"
LABEL="Nominal start and overhead, ideal conditions, run at $(date --iso=s)"
date --iso=s
run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 3 "${OPSIMRUN}" \
  --keep_rewards --label "${LABEL}" --archive ${ARCHIVE} --capture_env \
  --delay 0 --anom_overhead_scale 0 \
  --tags ideal nominal


for DELAY in 10 60 ; do
  echo "Running SV simulation delayed ${DELAY}"
  OPSIMRUN="prenight_delay${DELAY}_$(date --iso=s)"
  LABEL="Start time delayed by ${DELAY} minutes, nominal slew and visit overhead, ideal conditions, run at $(date --iso=s)"
  date --iso=s
  run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 3 "${OPSIMRUN}" \
    --label "${LABEL}" --archive ${ARCHIVE} --capture_env \
    --delay ${DELAY} --anom_overhead_scale 0 \
    --tags ideal delay_${DELAY}
done

ANOM_SCALE="0.1"
for ANOM_SEED in 101 102 ; do
  echo "Running SV simulation with anomalous overhead seed ${ANOM_SEED}"
  OPSIMRUN="prenight_anom${ANOM_SEED}_$(date --iso=s)"
  LABEL="Anomalous overhead (${ANOM_SEED}, ${ANOM_SCALE}), nominal start, ideal conditions, run at $(date --iso=s)"
  date --iso=s
  run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 3 "${OPSIMRUN}" \
    --label "${LABEL}" --archive ${ARCHIVE} --capture_env \
    --delay 0 --anom_overhead_scale ${ANOM_SCALE} --anom_overhead_seed ${ANOM_SEED} \
    --tags ideal anomalous_overhead
done
