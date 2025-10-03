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
LATEST_TAGGED_STACK=$(
    find /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib \
        -maxdepth 1 \
        -regex '.*/v[0-9]+\.[0-9]+\.[0-9]+' \
        -printf "%f\n" |
    sort -V |
    tail -1
)
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

if false ; then
  # Get latest tagged versions of everything
  RUBIN_SCHEDULER_REFERENCE=$(curl -s https://api.github.com/repos/lsst/rubin_scheduler/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
  RUBIN_SIM_REFERENCE=$(curl -s https://api.github.com/repos/lsst/rubin_sim/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
  SCHEDVIEW_REFERENCE=$(curl -s https://api.github.com/repos/lsst/schedview/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
  TS_FBS_UTILS_REFERENCE=$(curl -s https://api.github.com/repos/lsst-ts/ts_fbs_utils/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
  SIMS_SV_SURVEY_REFERENCE=$(curl -s https://api.github.com/repos/lsst-sims/sims_sv_survey/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
  RUBIN_NIGHTS_REFERENCE=$(curl -s https://api.github.com/repos/lsst-sims/rubin_nights/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
else
  # alternately, set specific versions
  RUBIN_SCHEDULER_REFERENCE="v3.14.1"
  RUBIN_SIM_REFERENCE="tickets/SP-2167"
  SCHEDVIEW_REFERENCE="main"
  TS_FBS_UTILS_REFERENCE="v0.17.0"
  SIMS_SV_SURVEY_REFERENCE="tickets/SP-2167"
  RUBIN_NIGHTS_REFERENCE="v0.6.1"
fi

pip install --no-deps --target=${PACKAGE_DIR} \
  git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_REFERENCE} \
  git+https://github.com/lsst/rubin_sim.git@${RUBIN_SIM_REFERENCE} \
  git+https://github.com/lsst/schedview.git@${SCHEDVIEW_REFERENCE} \
  git+https://github.com/lsst-ts/ts_fbs_utils.git@${TS_FBS_UTILS_REFERENCE} \
  git+https://github.com/lsst-sims/sims_sv_survey.git@${SIMS_SV_SURVEY_REFERENCE} \
  git+https://github.com/lsst-sims/rubin_nights.git@${RUBIN_NIGHTS_REFERENCE} \
  lsst-resources

if false ; then
  # Get the scheduler version from the EFD and install it.
  # We have to do this after the others, because we want
  # the version of obs_version_at_time supplied by the
  # version of schedview we specify.
  # Go ahead and install any missing dependencies as well.
  RUBIN_SCHEDULER_REFERENCE=v$(obs_version_at_time rubin_scheduler)
  echo "Using rubin_scheduler $RUBIN_SCHEDULER_REFERENCE"
  pip install --ignore-installed --no-deps --upgrade --target=${PACKAGE_DIR} git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_REFERENCE}
fi

# Get the scheduler configuration script
# It lives in ts_ocs_config
TS_CONFIG_OCS_REFERENCE="v0.28.33"
echo "Using ts_config_ocs ${TS_CONFIG_OCS_REFERENCE}"
SCHED_CONFIG_URL="https://raw.githubusercontent.com/lsst-ts/ts_config_ocs/refs/tags/${TS_CONFIG_OCS_REFERENCE}/Scheduler/feature_scheduler/maintel/fbs_config_sv_survey.py"
SCHED_CONFIG_FNAME=$(basename "$SCHED_CONFIG_URL")
curl -sL ${SCHED_CONFIG_URL} -o ${SCHED_CONFIG_FNAME}

# This config script also needs a supporting file:
SUPP_CONFIG_URL="https://raw.githubusercontent.com/lsst-ts/ts_config_ocs/refs/tags/${TS_CONFIG_OCS_REFERENCE}/Scheduler/feature_scheduler/maintel/ddf_sv.dat"
SUPP_CONFIG_FNAME=$(basename "${SUPP_CONFIG_URL}")
curl -sL ${SUPP_CONFIG_URL} -o ${SUPP_CONFIG_FNAME}

export DAYOBS="$(date -u --date='-12 hours' +'%Y%m%d')"
export LASTNIGHTISO="$(date --date='-36 hours' -u +'%F')"

export ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/test/"
export VSARCHIVE_PGDATABASE="opsim_log"
export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
export VSARCHIVE_PGUSER="tester"
export VSARCHIVE_PGSCHEMA="test"

echo "Fetching completed visits"
date --iso=s
fetch_sv_visits ${DAYOBS} completed_visits.db ~/.lsst/usdf_access_token

# Recording hash of fetched visits
COMPLETED=$(vseqarchive record-visitseq-metadata \
    completed \
    completed_visits.db \
    "Consdb query through ${LASTNIGHTISO}" \
    --first_day_obs 20250620 \
    --last_day_obs ${LASTNIGHTISO})

echo "Creating scheduler pickle"
date --iso=s
make_sv_scheduler scheduler.p --opsim completed_visits.db --config-script ${SCHED_CONFIG_FNAME}

echo "Creating model observatory"
date --iso=s
make_model_observatory observatory.p

# make dir for output
OPSIM_RESULT_DIR=${WORK_DIR}/opsim_results
mkdir ${OPSIM_RESULT_DIR}

echo "Running nominal SV simulation"
OPSIMRUN="prenight_nominal_$(date --iso=s)"
LABEL="Nominal start and overhead, ideal conditions, run at $(date --iso=s)"
date --iso=s
run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 1 "${OPSIMRUN}" \
  --keep_rewards --label "${LABEL}" \
  --delay 0 --anom_overhead_scale 0 \
  --results ${OPSIM_RESULT_DIR}

echo "Creating entry in metadatdata database"
date --iso=s
SIM_UUID=$(vseqarchive record-visitseq-metadata \
    simulations \
    ${OPSIM_RESULT_DIR}/opsim.db \
    "${LABEL}" \
    --first_day_obs ${DAYOBS} \
    --last_day_obs ${DAYOBS}
    )
vseqarchive update-visitseq-metadata ${SIM_UUID} parent_visitseq_uuid ${COMPLETED}
vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs ${LASTNIGHTISO}

vseqarchive update-visitseq-metadata ${SIM_UUID} scheduler_version "${RUBIN_SCHEDULER_REFERENCE}"
vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}
vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}
vseqarchive tag ${SIM_UUID} test prenight ideal nominal

CONDA_HASH=$(vseqarchive record-conda-env)
vseqarchive update-visitseq-metadata ${SIM_UUID} conda_env_sha256 ${CONDA_HASH}

vseqarchive get-file ${SIM_UUID} visits visits.h5
vseqarchive add-nightly-stats ${SIM_UUID} visits.h5 azimuth altitude

rm visits.h5 ${OPSIM_RESULT_DIR}/opsim.db ${OPSIM_RESULT_DIR}/rewards.h5 ${OPSIM_RESULT_DIR}/obs_stats.txt ${OPSIM_RESULT_DIR}/observatory.p ${OPSIM_RESULT_DIR}/scheduler.p ${OPSIM_RESULT_DIR}/sim_metadata.yaml

for DELAY in 60 240 ; do
  echo "Running SV simulation delayed ${DELAY}"
  OPSIMRUN="prenight_delay${DELAY}_$(date --iso=s)"
  LABEL="Start time delayed by ${DELAY} minutes, nominal slew and visit overhead, ideal conditions, run at $(date --iso=s)"
  date --iso=s
  run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 1 "${OPSIMRUN}" \
    --keep_rewards --label "${LABEL}" \
    --delay ${DELAY} --anom_overhead_scale 0 \
    --results ${OPSIM_RESULT_DIR}

  SIM_UUID=$(vseqarchive record-visitseq-metadata \
      simulations \
      ${OPSIM_RESULT_DIR}/opsim.db \
      "${LABEL}" \
      --first_day_obs ${DAYOBS} \
      --last_day_obs ${DAYOBS}
      )
  vseqarchive update-visitseq-metadata ${SIM_UUID} parent_visitseq_uuid ${COMPLETED}
  vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs ${LASTNIGHTISO}
  vseqarchive update-visitseq-metadata ${SIM_UUID} scheduler_version "${RUBIN_SCHEDULER_REFERENCE}"
  vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}
  vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}
  vseqarchive tag ${SIM_UUID} test prenight ideal delay_${DELAY}
  vseqarchive update-visitseq-metadata ${SIM_UUID} conda_env_sha256 ${CONDA_HASH}
  vseqarchive get-file ${SIM_UUID} visits visits.h5
  vseqarchive add-nightly-stats ${SIM_UUID} visits.h5 azimuth altitude

  rm visits.h5 ${OPSIM_RESULT_DIR}/opsim.db ${OPSIM_RESULT_DIR}/rewards.h5 ${OPSIM_RESULT_DIR}/obs_stats.txt ${OPSIM_RESULT_DIR}/observatory.p ${OPSIM_RESULT_DIR}/scheduler.p ${OPSIM_RESULT_DIR}/sim_metadata.yaml
done

ANOM_SCALE="0.1"
for ANOM_SEED in 101 102 ; do
  echo "Running SV simulation with anomalous overhead seed ${ANOM_SEED}"
  OPSIMRUN="prenight_anom${ANOM_SEED}_$(date --iso=s)"
  LABEL="Anomalous overhead (${ANOM_SEED}, ${ANOM_SCALE}), nominal start, ideal conditions, run at $(date --iso=s)"
  date --iso=s
  run_sv_sim scheduler.p observatory.p "" ${DAYOBS} 1 "${OPSIMRUN}" \
    --keep_rewards --label "${LABEL}" \
    --delay 0 --anom_overhead_scale ${ANOM_SCALE} \
    --results ${OPSIM_RESULT_DIR}

  SIM_UUID=$(vseqarchive record-visitseq-metadata \
      simulations \
      ${OPSIM_RESULT_DIR}/opsim.db \
      "${LABEL}" \
      --first_day_obs ${DAYOBS} \
      --last_day_obs ${DAYOBS}
      )
  vseqarchive update-visitseq-metadata ${SIM_UUID} parent_visitseq_uuid ${COMPLETED}
  vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs ${LASTNIGHTISO}
  vseqarchive update-visitseq-metadata ${SIM_UUID} scheduler_version "${RUBIN_SCHEDULER_REFERENCE}"
  vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}
  vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}
  vseqarchive tag ${SIM_UUID} test prenight ideal anomalous_overhead
  vseqarchive update-visitseq-metadata ${SIM_UUID} conda_env_sha256 ${CONDA_HASH}
  vseqarchive get-file ${SIM_UUID} visits visits.h5
  vseqarchive add-nightly-stats ${SIM_UUID} visits.h5 azimuth altitude

  rm visits.h5 ${OPSIM_RESULT_DIR}/opsim.db ${OPSIM_RESULT_DIR}/rewards.h5 ${OPSIM_RESULT_DIR}/obs_stats.txt ${OPSIM_RESULT_DIR}/observatory.p ${OPSIM_RESULT_DIR}/scheduler.p ${OPSIM_RESULT_DIR}/sim_metadata.yaml
done
