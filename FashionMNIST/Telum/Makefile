classifier.jar: classifier/fashion_mnist.class
	jar -cf classifier.jar classifier

imageio.jar: imageio/pgm.class imageio/tests/test_pgm.class
	jar -cf imageio.jar

classifier/fashion_mnist.class: classifier/fashion_mnist.java
	javac -classpath c_java_libs/lib/javaruntime.jar:.  classifier/fashion_mnist.java

imageio/pgm.class: imageio/pgm.java
	javac imageio/pgm.java

imageio/test/test_pgm.class: imageio/tests/test_pgm.java
	javac imageio/tests/test_pgm.java