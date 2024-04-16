# Code for inferencing on the z16/Telum based systems

`classifier.py` -> Classifier script, follow steps below.

To do the FashionMNIST example, first train it on another system (e.g. Graphcore) and copy the ONNX model to your mainframe. Make sure you have podman/docker installed (and signed in to IBM's container registry) and then to compile the model:

```
podman run --rm -v $(pwd):/workdir:z icr.io/ibmz/zdlc:4.1.1 --EmitLib --O3 --mcpu=z16 --mtriple=s390x-ibm-loz --maccel=NNPA fashion_classifier.onnx
mkdir fashion_MNIST
cp fashion_classifier.so fashion_MNIST
```

(adjusting names of things as appropriate)

You'll need to pull the libraries out of the ZDLC image and put them somewhere in your `PYTHONPATH`

```
mkdir pylibs
podman run --rm -v $(pwd)/pylibs:/files:z --entrypoint '/usr/bin/bash' icr.io/ibmz/zdlc:4.1.1 -c "cp /usr/local/lib/PyRuntime.cpython-*-s390x-linux-gnu.so /files"
export PYTHONPATH=$(pwd)/pylibs:${PYTHONPATH}
```

You should then be able to run `classifier.py` with a 28x28*256 greys pgm image of clothing (e.g. extracted from the FashionMNIST set)
