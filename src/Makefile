CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra 
LIBS = -lm
NVLIBS = -L/usr/local/cuda/lib64 -lcudadevrt -lcublas -arch=sm_50
PROGS = test

all: $(PROGS)

test: test.o matrix.o qr.o gpu_qr.o
	nvcc -G -o test test.o matrix.o qr.o gpu_qr.o $(LIBS) $(NVLIBS)

matrix.o: matrix.c
	$(CC) -c matrix.c

test.o: test.c
	$(CC) -c test.c

qr.o: qr.c
	$(CC) -c qr.c

gpu_qr.o: gpu_qr.cu
	$(NVCC) -dc -G gpu_qr.cu 

clean:
	rm -f *.o $(PROGS) 
