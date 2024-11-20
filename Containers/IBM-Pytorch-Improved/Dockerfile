FROM icr.io/ibmz/ibmz-accelerated-for-pytorch:1.1.0

USER root

RUN apt-get update && apt-get dist-upgrade -y && apt-get autoremove -y && apt-get install -y git python3.10-venv vim tmux screen

USER ibm-user