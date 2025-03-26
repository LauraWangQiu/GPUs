
CXX = g++
CC = gcc
NVCC = nvcc
# NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`

CFLAGS=-O3 -I.
# CFLAGS=-O3 -I. -g -G
CXXFLAGS=-O3 -I.

LIBS =-lpng -lm -lcuda -lcudart

SRC = png_io.o routinesCPU.o routinesGPU.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) -allow-unsupported-compiler $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

image: $(SRC) 
	$(CXX) -o image  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o image
	rm -f out.png out_canny.png out_hough.png

cleanImages:
	rm -f out.png out_canny.png out_hough.png