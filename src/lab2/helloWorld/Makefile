CXX=icpx
# CXXFLAGS=-fsycl -I.
CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -I. # Specify the default_selector
LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
