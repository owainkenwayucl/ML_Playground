# Graphcore FashionMNIST PopTorch training code.

This code is basically https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch/basics/README.html wrapped up into a script. It also dumps the model as a ONNX file for moving to other systems for inference.

When run it outputs two files:

`fashion_classifier.pth` -> Pytorch version of the model.
`fashion_classifier.onnx` -> ONNX version.