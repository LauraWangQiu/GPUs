CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
CXX = g++
CC = g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"

CFLAGS=-O3 -I. -I$(CUDA_PATH)/include
CXXFLAGS=-O3 -I. -I$(CUDA_PATH)/include

LIBS = -L$(CUDA_PATH)/lib/x64 -lcuda -lcudart -lstdc++ -lmsvcrt -lucrt

SRC =  main.o matrix_mul.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

matrix_mul: $(SRC) 
	$(CC) -o matrix_mult.exe  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	del /Q *.o matrix_mult.exe
