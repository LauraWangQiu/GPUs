#include <sycl/sycl.hpp>

using  namespace  sycl;

int main(int argc, char **argv) {

	if (argc!=2)  {
	std::cout << "./exec N"<< std::endl;
	return(-1);
	}

	int N = atoi(argv[1]);

	sycl::queue Q(sycl::default_selector_v);

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;


	int *a = (int*)malloc_shared(N * sizeof(int), Q);
	int *b = (int*)malloc_shared(N * sizeof(int), Q);
	int *c = (int*)malloc_shared(N * sizeof(int), Q);

	// Parallel for
	for(int i=0; i<N; i++){
		a[i] = i;   // Init a
		b[i] = i*i; // Init b
	}

	// Create a kernel to perform c=a+b
	Q.submit([&](handler &h) { 
		h.parallel_for(N, [=](id<1> i) {
			c[i] = a[i] + b[i];
		});
	}).wait();

	for(int i=0; i<N; i++)
		std::cout << "c[" << i << "] = " << c[i] << std::endl;

	free(a, Q);
	free(b, Q);
	free(c, Q);

	return 0;
}
