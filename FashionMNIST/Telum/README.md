# Code for inferencing on the z16/Telum based systems

## Python

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

## Java

`classifier.java` -> Classifier class, follow steps below.

To do the FashionMNIST example, first train it on another system (e.g. Graphcore) and copy the ONNX model to your mainframe. Make sure you have podman/docker installed (and signed in to IBM's container registry) and then to compile the model:

```
podman run --rm -v $(pwd):/workdir:z icr.io/ibmz/zdlc:4.1.1 --EmitJNI --O3 --mcpu=z16 --mtriple=s390x-ibm-loz --maccel=NNPA fashion_classifier.onnx
mkdir fashion_MNIST
cp fashion_classifier.jar fashion_MNIST
```

(adjusting names of things as appropriate).

As with Python, you need to pull some libs out of the ZDLC image:

```
mkdir c_java_libs
podman run --rm -v $(pwd)/c_java_libs:/files:z --entrypoint '/usr/bin/bash' icr.io/ibmz/zdlc:4.1.1 -c "cp -r /usr/local/{include,lib} /files"
```

Then you need to compile the Java code.

```
javac -classpath c_java_libs/lib/javaruntime.jar:.  classifier.java
```

And to run it:

```
java -classpath fashion_MNIST/fashion_classifier.jar:. classifier fashion_MNIST/data/test/0.pgm
```

## Clojure

Follow the build steps for Java and then run `make` to build `.jar` file versions of `classifier` and `imagio` (make sure they match what's in `deps.edn`).

You should then be able to call it from clojure e.g.

```
[uccaoke@vind1 Telum]$ clj
Clojure 1.11.2
user=> (import imageio.pgm)
imageio.pgm
user=> (import classifier.fashion_mnist)
classifier.fashion_mnist
user=> (def image (new pgm))
#'user/image
user=> (.read_image image "test.pgm")
nil
user=> (.get_classification image)
"Ankle boot"
user=> (fashion_mnist/classify (fashion_mnist/inference image))
"Ankle boot"
user=> 
```