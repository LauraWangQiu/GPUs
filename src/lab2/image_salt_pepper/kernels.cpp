#include <math.h>

#include <sycl/sycl.hpp>

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float threshold, int window_size,
	int height, int width)
{
	int ws2 = (window_size - 1) >> 1;

    Q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        if (i >= ws2 && i < height - ws2 && j >= ws2 && j < width - ws2) {
            float window[25]; // Asume que el tamano maximo de la ventana es 5x5
            float median;

            for (int ii = -ws2; ii <= ws2; ii++) {
                for (int jj = -ws2; jj <= ws2; jj++) {
                    window[(ii + ws2) * window_size + (jj + ws2)] = im[(i + ii) * width + (j + jj)];
                }
            }

            for (int k = 0; k < window_size * window_size - 1; k++) {
                for (int l = k + 1; l < window_size * window_size; l++) {
                    if (window[k] > window[l]) {
                        float temp = window[k];
                        window[k] = window[l];
                        window[l] = temp;
                    }
                }
            }

            median = window[(window_size * window_size - 1) >> 1];

            if (fabsf((median - im[i * width + j]) / median) <= threshold)
                image_out[i * width + j] = im[i * width + j];
            else
                image_out[i * width + j] = median;
        }
    }).wait();
}
