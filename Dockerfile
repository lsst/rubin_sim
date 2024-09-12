# Follow https://micromamba-docker.readthedocs.io/en/latest/

# Base container
FROM mambaorg/micromamba:1.5.9

# Add rubin-sim from conda-forge

RUN micromamba install -y -n base -c conda-forge rubin-sim
RUN micromamba install -y -n base -c conda-forge \
       jinja2 \
       tornado
RUN micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Container execution
# Mount fbs simulation outputs expected at /data/fbs_sims

EXPOSE 8080
ENV PORT 8080

# Start up show_maf on port 8080
CMD cd /data/fbs_sims && show_maf -p 80 --no_browser
