CC=nvcc
RM=rm -f
BLAS=-lcublas
SRC1=native_mv_mult.cu
SRC2=cublas_mv_mult.cu
EXEC1=native_mv_mult
EXEC2=cublas_mv_mult

all: native_mv cublas_mv

native_mv:
	$(CC) src/$(SRC1) -o bin/$(EXEC1)

cublas_mv:
	$(CC) $(BLAS) src/$(SRC2) -o bin/$(EXEC2) 

.PHONY: clean

clean:
	$(RM) bin/*
