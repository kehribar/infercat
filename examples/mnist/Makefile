# -----------------------------------------------------------------------------
# 
# 
# -----------------------------------------------------------------------------

all:
	gcc main.c ../../src/infercat.c mnistReader.c -O3 -I ../../src -o main -lm

clean:
	rm main

generate_model:
	python3 ../../utils/model_export.py ./training/mnist_model mnist_model.h
