CXX=icpx
# CXXFLAGS=-fsycl -I.
# CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -I. 			# Specify the default_selector
CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -I. 	# Specify the default_selector and the posibility of choosing when executing
# ONEAPI_DEVICE_SELECTOR=opencl:cpu ./exec 
# ONEAPI_DEVICE_SELECTOR=opencl:gpu ./exec 

LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
