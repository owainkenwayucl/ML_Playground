FROM icr.io/ibmz/ibmz-accelerated-for-pytorch:1.2.0

USER root

# Update things
RUN apt-get update && apt-get dist-upgrade -y && apt-get autoremove -y

# Install virtualenv
RUN apt-get install -y git python3.10-venv

# Deps for torchvision
RUN apt-get install -y libpng-dev libpng++-dev libpng-tools libjpeg-turbo8-dev 

# Install user friendliness tools
RUN apt-get install -y vim tmux screen gfortran cmake make

USER ibm-user