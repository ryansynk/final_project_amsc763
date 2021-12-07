CC = gcc
CFLAGS = -Wall
LIBS = -lm
PROGS = test

all: $(PROGS)

test: test.o matrix.o
	$(CC) -o test test.o matrix.o $(LIBS)	

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c matrix.c

test.o: test.c test.h matrix.h
	$(CC) $(CFLAGS) -c test.c

clean:
	rm -f *.o $(PROGS)
