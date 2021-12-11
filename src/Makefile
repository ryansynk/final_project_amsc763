CC = gcc
CFLAGS = -Wall -Wextra -g
LIBS = -lm
PROGS = test

all: $(PROGS)

test: test.o matrix.o qr.o
	$(CC) -o test test.o matrix.o qr.o $(LIBS)	

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c matrix.c

test.o: test.c test.h matrix.h
	$(CC) $(CFLAGS) -c test.c

qr.o: qr.c qr.h
	$(CC) $(CFLAGS) -c qr.c

clean:
	rm -f *.o $(PROGS) 
