#include <CL/sycl.hpp>

using  namespace  cl::sycl;

#define VERSION_1 // Using accesors to read buffer_a
// #define VERSION_2 // Using {} to delete buffer_a

int main(int argc, char **argv) {

    if (argc!=2)  {
		std::cout << "./exec N"<< std::endl;
		return(-1);
	}

	int N = atoi(argv[1]);

	sycl::queue Q(sycl::gpu_selector_v);

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	std::vector<float> a(N);

    for(int i=0; i<N; i++)
        a[i] = i; // Init a

    #ifdef VERSION_2
    {
    #endif
    //Create a submit a kernel
    buffer buffer_a{a}; //Create a buffer with values of array a

    // Create a command_group to issue command to the group
    Q.submit([&](handler &h) {
        accessor acc_a{buffer_a, h, read_write}; // Accessor to buffer_a

        // Submit the kernel
        h.parallel_for(N, [=](id<1> i) {
            acc_a[i]*=2.0f;
        }); // End of the kernel function
    }).wait();       // End of the queue commands we waint on the event reported.
    #ifdef VERSION_2
    }
    #elif defined VERSION_1
    host_accessor a_(buffer_a,read_only);
    #endif

    for(int i=0; i<N; i++)
        std::cout << "a[" << i << "] = " << a[i] << std::endl;

  return 0;
}
