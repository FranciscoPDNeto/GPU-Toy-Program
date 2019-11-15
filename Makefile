CC := g++
CFLAGS := -Wall -O3 -std=c++17 -lOpenCL

all: gpuTest

gpuTest: gpuTest.o
	$(CC) $(CFLAGS) -o $@ $^
	rm -rf gpuTest.o

gpuTest.o: gpuTest.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm gpuTest