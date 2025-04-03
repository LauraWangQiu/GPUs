#ifndef KERNEL_H 
#define KERNEL_H 

#include <sycl/sycl.hpp>

using namespace sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float threshold, int window_size,
	int height, int width);
#endif
