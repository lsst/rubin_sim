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

set -o xtrace

export AWS_PROFILE=prenight
WORK_DATE=$(date '+%Y-%m-%dT%H%M%S' --utc)
WORK_DIR="/sdf/data/rubin/shared/scheduler/prenight/work/run_auxtel_prenight_sims/${WORK_DATE}"
echo "Working in $WORK_DIR"
mkdir ${WORK_DIR}
cd ${WORK_DIR}

# Install required python packages in a new venv
# The true environment is from the venv created below,
# but we need a python in the path to create the
# venv to begin with, so we source loadLSST.sh.
source /sdf/group/rubin/sw/w_latest/loadLSST.sh
PRENIGHT_VENV=$(mktemp -d /sdf/scratch/users/n/${USER}/prenight_venvs/prenight-${WORK_DATE}-XXXXXX)
python -m venv "${PRENIGHT_VENV}"
ln -s "${PRENIGHT_VENV}" "${WORK_DIR}/venv"
source "${PRENIGHT_VENV}/bin/activate"

if false ; then
  # Get latest tagged version lsst_survey_sim (and its dependencies)
  LSST_SURVEY_SIM_REFERENCE=$(curl -s https://api.github.com/repos/lsst-sims/lsst_survey_sim/tags | jq -r '.[].name' | egrep '^v[0-9]+.[0-9]+.[0-9]+$' | sort -V | tail -1)
else
  # alternately, use the HEAD of main
  LSST_SURVEY_SIM_REFERENCE="main"
fi

pip install \
  git+https://github.com/lsst-sims/lsst_survey_sim.git@${LSST_SURVEY_SIM_REFERENCE} \
  click \
  psycopg2 \
  botocore \
  boto3 \
  lsst-resources

if false ; then
  # Get the scheduler version from the EFD and install it.
  # We have to do this after the others, because we want
  # the version of obs_version_at_time supplied by the
  # version of schedview we specify.
  # Go ahead and install any missing dependencies as well.
  RUBIN_SCHEDULER_REFERENCE=v$(obs_version_at_time rubin_scheduler)
  echo "Using rubin_scheduler $RUBIN_SCHEDULER_REFERENCE"
  pip install --ignore-installed --no-deps --upgrade git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_REFERENCE}
fi

# Get the scheduler configuration script
# It lives in ts_config_scheduler
TS_CONFIG_SCHEDULER_REFERENCE="develop"
SCHED_CONFIG_FNAME="ts_config_scheduler/Scheduler/feature_scheduler/auxtel/fbs_spec_flex_survey.py"
echo "Using ts_config_scheduler ${SCHED_CONFIG_FNAME} from ${TS_CONFIG_SCHEDULER_REFERENCE}"
git clone --depth 1 https://github.com/lsst-ts/ts_config_scheduler
cd ts_config_scheduler
git fetch --depth 1 origin "${TS_CONFIG_SCHEDULER_REFERENCE}"
git checkout FETCH_HEAD
cd ${WORK_DIR}

export DAYOBS="$(date -u --date='-12 hours' +'%Y%m%d')"
export NEXT_DAYOBS="$(date -u --date='+12 hours' +'%Y%m%d')"
export LAST_DAYOBS="$(date -u --date='+36 hours' +'%Y%m%d')"
export DAYOBS_SIMULATED="$DAYOBS $NEXT_DAYOBS $LAST_DAYOBS"
export LASTNIGHTISO="$(date --date='-36 hours' -u +'%F')"

export ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/vseq/"
export VSARCHIVE_PGDATABASE="opsim_log"
export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
export VSARCHIVE_PGUSER="writer"
export VSARCHIVE_PGSCHEMA="vsmd"

# Get an empty set of completed visits so we have something
# to pass make_lsst_scheduler
fetch_lsst_visits 20000101 completed_visits.db ~/.lsst/usdf_access_token

echo "Creating scheduler pickle"
date --iso=s
make_lsst_scheduler scheduler.p --opsim completed_visits.db --config_script ${SCHED_CONFIG_FNAME}

echo "Creating model observatory"
date --iso=s
ideal_model_observatory scheduler.p observatory.p

# make dir for output
OPSIM_RESULT_DIR=${WORK_DIR}/opsim_results
mkdir ${OPSIM_RESULT_DIR}

echo "Running nominal LSST simulation"
OPSIMRUN="prenight_nominal_$(date --iso=s)"
LABEL="Nominal start and overhead, ideal conditions, run at $(date --iso=s)"
date --iso=s
run_lsst_sim scheduler.p observatory.p "" ${DAYOBS} 3 "${OPSIMRUN}" \
  --keep_rewards --label "${LABEL}" \
  --delay 0 --anom_overhead_scale 0 \
  --results ${OPSIM_RESULT_DIR}


echo "Creating entry in metadatdata database"
date --iso=s
SIM_UUID=$(vseqarchive record-visitseq-metadata \
    simulations \
    ${OPSIM_RESULT_DIR}/opsim.db \
    "${LABEL}" \
    --telescope auxtel \
    --first_day_obs ${DAYOBS} \
    --last_day_obs ${LAST_DAYOBS}
    )

vseqarchive update-visitseq-metadata ${SIM_UUID} scheduler_version "${RUBIN_SCHEDULER_REFERENCE}"
vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}
vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}
vseqarchive tag ${SIM_UUID} prenight ideal nominal

CONDA_HASH=$(vseqarchive record-conda-env)
vseqarchive update-visitseq-metadata ${SIM_UUID} conda_env_sha256 ${CONDA_HASH}

vseqarchive get-file ${SIM_UUID} visits visits.h5
vseqarchive add-nightly-stats ${SIM_UUID} visits.h5 azimuth altitude

rm visits.h5 ${OPSIM_RESULT_DIR}/opsim.db ${OPSIM_RESULT_DIR}/rewards.h5 ${OPSIM_RESULT_DIR}/obs_stats.txt ${OPSIM_RESULT_DIR}/observatory.p ${OPSIM_RESULT_DIR}/scheduler.p ${OPSIM_RESULT_DIR}/sim_metadata.yaml

for DAYOBS_TO_INDEX in ${DAYOBS_SIMULATED}; do
  vseqarchive make-prenight-index ${DAYOBS_TO_INDEX} auxtel
done
