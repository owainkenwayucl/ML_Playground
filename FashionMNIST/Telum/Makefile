jars: classifier.jar imageio.jar Makefile

classifier.jar: classifier/fashion_mnist.class Makefile
	jar -cf classifier.jar classifier

imageio.jar: imageio/pgm.class imageio/tests/test_pgm.class Makefile 
	jar -cf imageio.jar

classifier/fashion_mnist.class: classifier/fashion_mnist.java Makefile 
	javac -classpath c_java_libs/lib/javaruntime.jar:.  classifier/fashion_mnist.java

imageio/pgm.class: imageio/pgm.java Makefile
	javac imageio/pgm.java

imageio/tests/test_pgm.class: imageio/tests/test_pgm.java Makefile
	javac imageio/tests/test_pgm.java

clean:
	rm -f imageio/*.class
	rm -f imageio/tests/*.class
	rm -f classifier/*.class
	rm -f classifier.jar imageio.jar 

.PHONY: clean jars