# Follow https://micromamba-docker.readthedocs.io/en/latest/

# Base container
FROM mambaorg/micromamba:1.5.9

# Copy current directory
COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/${MAMBA_USER}/rubin_sim

# Install container requirements from conda-forge
# Note that open-orb dependencies are omitted
RUN micromamba install -y -n base -f /home/${MAMBA_USER}/rubin_sim/container_environment.yaml
RUN micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install current version of rubin-sim
RUN python -m pip install /home/$MAMBA_USER/rubin_sim --no-deps

# Container execution
# Mount fbs simulation outputs expected at /data/fbs_sims
# Mount rubin_sim_data (if needed) at /data/rubin_sim_data

EXPOSE 8080
ENV PORT=8080

ENV RUBIN_SIM_DATA_DIR=/data/rubin_sim_data

# Start up show_maf on port 8080
CMD cd /data/fbs_sims && show_maf -p 8080 --no_browser
