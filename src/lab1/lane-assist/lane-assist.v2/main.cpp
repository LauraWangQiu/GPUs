#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "routinesCPU.h"
#include "routinesGPU.h"
#include "png_io.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

int main(int argc, char **argv)
{
	uint8_t *imgtmp, *imEdge;
	int width, height;

	int nlines = 0; 
	int x1[10], x2[10], y1[10], y2[10];
	double t0, t1;

	// Only accept a concrete number of arguments
	if (argc != 3)
	{
		printf("./exec image.png [c/g]\n");
		exit(-1);
	}

	// Read image
	imgtmp = read_png_fileRGB(argv[1], &width, &height);

	// Create temporal buffer
	imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	// Create the accumulators
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width  = 180;

	switch (argv[2][0])
	{
		case 'c':
			t0 = get_time();
			lane_assist_CPU(imgtmp, height, width, 
				imEdge, accu_height, accu_width,
				x1, y1, x2, y2, &nlines);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
			t0 = get_time();
			lane_assist_GPU(imgtmp, height, width,
				imEdge, accu_height, accu_width,
				x1, y1, x2, y2, &nlines);
            t1 = get_time();
			printf("GPU Exection time %f ms.\n", t1-t0);
			break;
		default:
			printf("Not Implemented yet!!\n");
	}

	for (int l=0; l<nlines; l++)
		printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

	// Export images
	write_png_fileBW("out_canny.png", imEdge, width, height);
	write_png_fileRGB("out.png", imgtmp, width, height);
}
