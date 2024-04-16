# Qualcomm Cloud Ai100 work

`classifier.py` -> Classifier script, follow steps below.

To do the FashionMNIST example, first train it on another system (e.g. Graphcore) and copy the ONNX model to your system with the Qualcomm Cloud Ai100 accelerators.

To compile the model you need to run:

```
/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0 -compile-only -aic-num-cores=4 -m=fashion_MNIST/fashion_classifier.onnx -onnx-define-symbol=batch_size,1 -aic-binary-dir=fashion_MNIST/aic-binary-dir
```

(where `fashion_MNIST/fashion_classifier.onnx` is your ONNX model)

You need to be in the group that can access the accelerators. On our system, that is `qaic`.

You need a Python 3.8.18. This is older than the version of Python that ships with RHEL 9.

Make sure to install `readline-devel` and `openssl-devel` before building it.

Freshly armed with  your new old python, you need to build a `qaic` wheel. You can find the source code hiding in `/opt`. You may think this means you can use a newer python but because a bunch of deps are binaries built for 3.8 you can't.

Once you have a python environment in which you can successfully run `import qaic` you can run `classifier.py`. It takes a single argument which is a 28x28x256 pgm file.