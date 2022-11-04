CC1=nvcc
CC2=g++
RM=rm -f
SRC1=fem_application.cu
SRC2=fem_application.cpp
EXEC1=app.cuda 
EXEC2=app.host
OPT=-O3
OMP=-fopenmp
ARCH1=sm_75
ARCH2=sandybridge

cuda: 
	$(CC1) -arch=$(ARCH1) src/$(SRC1) -o bin/$(EXEC1) 

gcc:
	$(CC2) $(OPT) -march=$(ARCH2) $(OMP) src/$(SRC2) -o bin/$(EXEC2) 

.PHONY: clean
clean:
	$(RM) bin/*
