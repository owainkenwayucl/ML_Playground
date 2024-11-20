FROM icr.io/ibmz/ibmz-accelerated-for-pytorch:1.1.0

USER root

# Update things
RUN apt-get update && apt-get dist-upgrade -y && apt-get autoremove -y

# Install virtualenv
RUN apt-get install -y git python3.10-venv 

# Install user friendliness tools
RUN vim tmux screen

USER ibm-user