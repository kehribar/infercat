# -----------------------------------------------------------------------------
# 
# 
# -----------------------------------------------------------------------------

all:
	gcc main.c ../../src/infercat.c mnistReader.c -I ../../src -o main -lm -O3

clean:
	rm main

generate_model:
	python3 ../../utils/model_export.py ./training/mnist_model mnist_model.h
