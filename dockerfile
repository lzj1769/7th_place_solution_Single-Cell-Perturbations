############################################################
# Dockerfile for Single Cell Perturbations 
# Based on Debian slim
############################################################

FROM ubuntu@sha256:2fdb1cf4995abb74c035e5f520c0f3a46f12b3377a59e86ecca66d8606ad64f9

LABEL maintainer = "Zhijian Li"

# To prevent time zone prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV RETICULATE_MINICONDA_ENABLED=FALSE

## Create new user 
ENV USER=scp
WORKDIR /home/$USER
RUN groupadd -r $USER && \
    useradd -r -g $USER --home /home/$USER -s /sbin/nologin -c "Docker image user" $USER &&\
    chown $USER:$USER /home/$USER

# Install libraries
RUN apt-get update
    
# Install python and R
RUN apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv

# COPY src/bash/monitor_script.sh /usr/local/bin

