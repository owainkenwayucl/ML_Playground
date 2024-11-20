# IBM-Pytorch-Improved

This container builds on the `icr.io/ibmz/ibmz-accelerated-for-pytorch:1.1.0` from [ibmz-accelerated-for-pytorch](https://github.com/IBM/ibmz-accelerated-for-pytorch) to add some extra tools - principally so that `python3 -m venv` and `pip` work properly.

You can find the wheels for PyTorch and `torch_nnpa` in `/pytorch/dist` and `/torch_nnpa` in the container image and using this you can build virtualenvs.