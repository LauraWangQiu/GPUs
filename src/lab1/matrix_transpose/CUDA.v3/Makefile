# CC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc
CC = nvcc

EXE   = transpose 

SOURCES    = transpose.cu


OBJS    = $(SOURCES:.cu=.o)

CFLAGS     = -O3  

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS) -allow-unsupported-compiler -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu
	$(CC) $(CFLAGS) -allow-unsupported-compiler -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)
