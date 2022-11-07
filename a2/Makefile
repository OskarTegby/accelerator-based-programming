CC=nvcc
RM=rm -f
BLAS=-lcublas
SRC1=native.cu
SRC2=cublas.cu
EXEC1=native
EXEC2=cublas

all: native cublas

native:
	$(CC) src/$(SRC1) -o bin/$(EXEC1)

cublas:
	$(CC) $(BLAS) src/$(SRC2) -o bin/$(EXEC2) 

.PHONY: clean

clean:
	$(RM) bin/*
