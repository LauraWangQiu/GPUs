#include <math.h>

#include <sycl/sycl.hpp>

#define MAX_WINDOW_SIZE 5 * 5

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float threshold, int window_size,
	int height, int width)
{
	int ws2 = (window_size - 1) >> 1;

    Q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
        int i, j, ii, jj;

        float window[MAX_WINDOW_SIZE];
        float median;

        i = idx[0];
        j = idx[1];

        if (i >= ws2 && i < height - ws2 && j >= ws2 && j < width - ws2) {
            for (ii = -ws2; ii <= ws2; ii++) {
                for (jj = -ws2; jj <= ws2; jj++) {
                    window[(ii + ws2) * window_size + (jj + ws2)] = im[(i + ii) * width + (j + jj)];
                }
            }

            // bubble_sort
            int k, l;
	        float tmp;

            for (k = 1; k < window_size * window_size; k++)
                for (l = 0; l < window_size * window_size - k; l++)
                    if (window[l] > window[l + 1])
                    {
                        tmp = window[l];
                        window[l] = window[l + 1];
                        window[l + 1] = tmp;
                    }
            median = window[(window_size * window_size - 1) >> 1];

            if (fabsf((median - im[i * width + j]) / median) <= threshold)
                image_out[i * width + j] = im[i * width + j];
            else
                image_out[i * width + j] = median;
        }
    }).wait();
}
