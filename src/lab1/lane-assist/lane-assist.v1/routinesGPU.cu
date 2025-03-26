#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"
#include "png_io.h"

#define DEG2RAD 0.017453f

#define BLOCK_SIZE 16

__global__ void init_cos_sin_table_kernel(float *sin_table, float *cos_table, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        sin_table[i] = sinf(i * DEG2RAD);
        cos_table[i] = cosf(i * DEG2RAD);
    }
}

__global__ void image_RGB2BW_kernel(uint8_t *image_in, uint8_t *image_out, int height, int width)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float R, G, B;

    if (i < height && j < width)
    {
        R = (float)(image_in[3 * (i * width + j)    ]);
        G = (float)(image_in[3 * (i * width + j) + 1]);
        B = (float)(image_in[3 * (i * width + j) + 2]);

        image_out[i * width + j] = (uint8_t)(0.2989 * R + 0.5870 * G + 0.1140 * B);
    }
}

__global__ void noiseReduction_kernel(uint8_t *im, float *NR, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Noise reduction
		NR[i * width + j] = (2.0 * im[(i - 2) * width + (j - 2)] + 4.0 * im[(i - 2) * width + (j - 1)] + 5.0 * im[(i - 2) * width + (j)] + 4.0 * im[(i - 2) * width + (j + 1)] + 2.0 * im[(i - 2) * width + (j + 2)] + 4.0 * im[(i - 1) * width + (j - 2)] + 9.0 * im[(i - 1) * width + (j - 1)] + 12.0 * im[(i - 1) * width + (j)] + 9.0 * im[(i - 1) * width + (j + 1)] + 4.0 * im[(i - 1) * width + (j + 2)] + 5.0 * im[(i)*width + (j - 2)] + 12.0 * im[(i)*width + (j - 1)] + 15.0 * im[(i)*width + (j)] + 12.0 * im[(i)*width + (j + 1)] + 5.0 * im[(i)*width + (j + 2)] + 4.0 * im[(i + 1) * width + (j - 2)] + 9.0 * im[(i + 1) * width + (j - 1)] + 12.0 * im[(i + 1) * width + (j)] + 9.0 * im[(i + 1) * width + (j + 1)] + 4.0 * im[(i + 1) * width + (j + 2)] + 2.0 * im[(i + 2) * width + (j - 2)] + 4.0 * im[(i + 2) * width + (j - 1)] + 5.0 * im[(i + 2) * width + (j)] + 4.0 * im[(i + 2) * width + (j + 1)] + 2.0 * im[(i + 2) * width + (j + 2)]) / 159.0;
	}
}

__global__ void gradient_kernel(float *NR, float *G, float *phi, float *Gx, float *Gy,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float PI = 3.141593;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Intensity gradient of the image
		Gx[i * width + j] =
			(1.0 * NR[(i - 2) * width + (j - 2)] + 2.0 * NR[(i - 2) * width + (j - 1)] + (-2.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + 4.0 * NR[(i - 1) * width + (j - 2)] + 8.0 * NR[(i - 1) * width + (j - 1)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-4.0) * NR[(i - 1) * width + (j + 2)] + 6.0 * NR[(i)*width + (j - 2)] + 12.0 * NR[(i)*width + (j - 1)] + (-12.0) * NR[(i)*width + (j + 1)] + (-6.0) * NR[(i)*width + (j + 2)] + 4.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + (-8.0) * NR[(i + 1) * width + (j + 1)] + (-4.0) * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 2.0 * NR[(i + 2) * width + (j - 1)] + (-2.0) * NR[(i + 2) * width + (j + 1)] + (-1.0) * NR[(i + 2) * width + (j + 2)]);

		Gy[i * width + j] =
			((-1.0) * NR[(i - 2) * width + (j - 2)] + (-4.0) * NR[(i - 2) * width + (j - 1)] + (-6.0) * NR[(i - 2) * width + (j)] + (-4.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + (-2.0) * NR[(i - 1) * width + (j - 2)] + (-8.0) * NR[(i - 1) * width + (j - 1)] + (-12.0) * NR[(i - 1) * width + (j)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-2.0) * NR[(i - 1) * width + (j + 2)] + 2.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + 12.0 * NR[(i + 1) * width + (j)] + 8.0 * NR[(i + 1) * width + (j + 1)] + 2.0 * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 4.0 * NR[(i + 2) * width + (j - 1)] + 6.0 * NR[(i + 2) * width + (j)] + 4.0 * NR[(i + 2) * width + (j + 1)] + 1.0 * NR[(i + 2) * width + (j + 2)]);

		G[i * width + j] = sqrtf((Gx[i * width + j] * Gx[i * width + j]) + (Gy[i * width + j] * Gy[i * width + j])); // G = √Gx²+Gy²
		phi[i * width + j] = atan2f(fabs(Gy[i * width + j]), fabs(Gx[i * width + j]));
		
		if (fabs(phi[i * width + j]) <= PI / 8)
			phi[i * width + j] = 0;
		else if (fabs(phi[i * width + j]) <= 3 * (PI / 8))
			phi[i * width + j] = 45;
		else if (fabs(phi[i * width + j]) <= 5 * (PI / 8))
			phi[i * width + j] = 90;
		else if (fabs(phi[i * width + j]) <= 7 * (PI / 8))
			phi[i * width + j] = 135;
		else
			phi[i * width + j] = 0;
	} 
}

__global__ void edge_kernel(float *G, float *phi, uint8_t *pedge, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= 3 && i < height - 3 && j >= 3 && j < width - 3)
    {
        // Edge
        pedge[i * width + j] = 0;
        if (phi[i * width + j] == 0)
        {
            if (G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1]) // edge is in N-S
                pedge[i * width + j] = 1;
        }
        else if (phi[i * width + j] == 45)
        {
            if (G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1]) // edge is in NW-SE
                pedge[i * width + j] = 1;
        }
        else if (phi[i * width + j] == 90)
        {
            if (G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j]) // edge is in E-W
                pedge[i * width + j] = 1;
        }
        else if (phi[i * width + j] == 135)
        {
            if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1]) // edge is in NE-SW
                pedge[i * width + j] = 1;
        }
    }
}

__global__ void thresholding_kernel(uint8_t *im, uint8_t *image_out,
	float *G, uint8_t *pedge, float level, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 3 && i < height - 3 && j >= 3 && j < width - 3)
	{
		// Hysteresis Thresholding
        float lowthres = level / 2;
        float hithres = 2 * (level);

        image_out[i * width + j] = 0;
        if (G[i * width + j] > hithres && pedge[i * width + j])
            image_out[i * width + j] = 255;
        else if (pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
            // check neighbours 3x3
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++)
                    if (G[(i + ii) * width + j + jj] > hithres)
                        image_out[i * width + j] = 255;
	}
}

__global__ void hough_kernel(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
	float *sin_table, float *cos_table, float hough_h, float center_x, float center_y)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < height && j < width)
	{
		if (im[(i * width) + j] > 250) // Pixel is edge
		{
			for (int theta = 0; theta < 180; theta++)
			{
				float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
				atomicAdd(&accumulators[(int)((round(rho + hough_h) * 180.0)) + theta], 1);
			}
		}
	}
}

__global__ void get_lines_kernel(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table, int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho = blockIdx.x * blockDim.x + threadIdx.x;
    int theta = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t max;

    if (rho < accu_height && theta < accu_width)
    {
        if (accumulators[(rho * accu_width) + theta] >= threshold)
        {
            // Is this point a local maxima (9x9)
            max = accumulators[(rho * accu_width) + theta];
            for (int ii = -4; ii <= 4; ii++)
			{
				for (int jj = -4; jj <= 4; jj++)
                {
                    if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width))
                    {
                        if (accumulators[((rho + ii) * accu_width) + (theta + jj)] > max)
                        {
                            max = accumulators[((rho + ii) * accu_width) + (theta + jj)];
                        }
                    }
                }
			}
			
            if (max == accumulators[(rho * accu_width) + theta]) // local maxima
            {
                int x1, y1, x2, y2;
                x1 = y1 = x2 = y2 = 0;

                if (theta >= 45 && theta <= 135)
                {
                    if (theta > 90)
                    {
                        // y = (r - x cos(t)) / sin(t)
                        x1 = width / 2;
                        y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
                        x2 = width;
                        y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
                    }
                    else
                    {
                        // y = (r - x cos(t)) / sin(t)
                        x1 = 0;
                        y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
                        x2 = width * 2 / 5;
                        y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
                    }
                }
                else
                {
                    // x = (r - y sin(t)) / cos(t);
                    y1 = 0;
                    x1 = ((float)(rho - (accu_height / 2)) - ((y1 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
                    y2 = height;
                    x2 = ((float)(rho - (accu_height / 2)) - ((y2 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
                }

                int idx = atomicAdd(lines, 1);
                x1_lines[idx] = x1;
                y1_lines[idx] = y1;
                x2_lines[idx] = x2;
                y2_lines[idx] = y2;
            }
        }
    }
}

__global__ void draw_lines_kernel(uint8_t *imgtmp, int width, int height, int *x1, int *y1, int *x2, int *y2, int nlines, int width_line)
{
	int l = blockIdx.x * blockDim.x + threadIdx.x;
    int wl = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < nlines && wl >= -(width_line >> 1) && wl <= (width_line >> 1))
    {	
        for (int x = x1[l]; x < x2[l]; x++)
        {
            int y = (float)(y2[l] - y1[l]) / (x2[l] - x1[l]) * (x - x1[l]) + y1[l]; // Line eq. known two points
            if (x + wl > 0 && x + wl < width && y > 0 && y < height)
            {
                imgtmp[3 * ((y) * width + x + wl)    ] = 255;
                imgtmp[3 * ((y) * width + x + wl) + 1] = 0;
                imgtmp[3 * ((y) * width + x + wl) + 2] = 0;
            }
        }
    }
}

void lane_assist_GPU(uint8_t *imgtmp, int height, int width, 
					 uint8_t *imEdge, int accu_height, int accu_width, 
					 int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	// Device variables 
	float *d_sin_table, *d_cos_table;
	int *d_x1, *d_y1, *d_x2, *d_y2, *d_nlines;
	uint8_t *d_image_in, *d_image_out, *d_imEdge;
    float *d_NR, *d_G, *d_phi, *d_Gx, *d_Gy;
    uint8_t *d_pedge;
    uint32_t *d_accum;

	cudaMalloc((void**)&d_sin_table, 	180 * 					   sizeof(float));
	cudaMalloc((void**)&d_cos_table, 	180 * 					   sizeof(float));
	cudaMalloc((void**)&d_x1, 			10  * 					   sizeof(int));
	cudaMalloc((void**)&d_y1, 			10  * 					   sizeof(int));
	cudaMalloc((void**)&d_x2, 			10  * 					   sizeof(int));
	cudaMalloc((void**)&d_y2, 			10  * 					   sizeof(int));
	cudaMalloc((void**)&d_nlines, 								   sizeof(int));
	cudaMalloc((void**)&d_image_in,	 	3 * height * width * 	   sizeof(uint8_t));
	cudaMalloc((void**)&d_image_out,	height * width * 		   sizeof(uint8_t));
    cudaMalloc((void**)&d_imEdge, 		height * width * 		   sizeof(uint8_t));
    cudaMalloc((void**)&d_NR, 			height * width * 		   sizeof(float));
    cudaMalloc((void**)&d_G, 			height * width * 		   sizeof(float));
    cudaMalloc((void**)&d_phi, 			height * width * 		   sizeof(float));
    cudaMalloc((void**)&d_Gx, 			height * width * 		   sizeof(float));
    cudaMalloc((void**)&d_Gy, 			height * width * 		   sizeof(float));
    cudaMalloc((void**)&d_pedge, 		height * width * 		   sizeof(uint8_t));
    cudaMalloc((void**)&d_accum, 		accu_width * accu_height * sizeof(uint32_t));

	// Initialize nlines to 0
	cudaMemset(d_nlines, 0, sizeof(int));
	// Initialize accumulators to 0
	cudaMemset(d_accum, 0, accu_width * accu_height * sizeof(uint32_t));
	
	// Copy image data to device
    cudaMemcpy(d_image_in, imgtmp, 3 * height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// Initialize sin and cos tables
	dim3 dimBlock(BLOCK_SIZE);
	int blocks = 180/BLOCK_SIZE;
	if (180%BLOCK_SIZE>0) blocks++;
	dim3 dimGrid(blocks);
    init_cos_sin_table_kernel<<<dimGrid, dimBlock>>>(d_sin_table, d_cos_table, 180);

	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dimGrid = dim3(ceil((float) width / dimBlock.x), ceil((float) height / dimBlock.y));
	image_RGB2BW_kernel<<<dimGrid, dimBlock>>>(d_image_in, d_image_out, height, width);
	
	// Canny (edge detection)
	// Split canny_kernel into 5 kernels
	int level = 1000.0f;
	noiseReduction_kernel<<<dimGrid, dimBlock>>>(d_image_out, d_NR, height, width);
	gradient_kernel		 <<<dimGrid, dimBlock>>>(d_NR, d_G, d_phi, d_Gx, d_Gy, height, width);
	edge_kernel			 <<<dimGrid, dimBlock>>>(d_G, d_phi, d_pedge, height, width);
	thresholding_kernel	 <<<dimGrid, dimBlock>>>(d_image_out, d_imEdge, d_G, d_pedge, level, height, width);
	
	// Copy edge image to host
	cudaMemcpy(imEdge, d_imEdge, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	// Hough transform (line detection)
	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);
    float center_x = width 	/ 2.0;
    float center_y = height / 2.0;
    hough_kernel<<<dimGrid, dimBlock>>>(d_imEdge, width, height, 
												   d_accum, accu_width, accu_height,
												   d_sin_table, d_cos_table, 
												   hough_h, center_x, center_y);
	
	int threshold;
	if (width > height) threshold = width / 6;
	else				threshold = height / 6;

    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dimGrid = dim3(ceil((float) accu_height / dimBlock.x), ceil((float) accu_width / dimBlock.y));
    get_lines_kernel<<<dimGrid, dimBlock>>>(threshold, d_accum, 
		accu_width, accu_height, 
		width, height, 
		d_sin_table, d_cos_table, 
		d_x1, d_y1, d_x2, d_y2, d_nlines);
	
    // Copy results back to host
    cudaMemcpy(x1, 		d_x1, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(y1, 		d_y1, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2, 		d_x2, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2, 		d_y2, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nlines, 	d_nlines,  sizeof(int), cudaMemcpyDeviceToHost);
	
	int width_line = 9;
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid = dim3(ceil((float) *nlines / dimBlock.x), ceil((float) width_line / dimBlock.y));
	draw_lines_kernel<<<dimGrid, dimBlock>>>(d_image_in, width, height,
		d_x1, d_y1, d_x2, d_y2, *nlines, width_line);
    
	cudaMemcpy(imgtmp, d_image_in, 3 * height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(d_sin_table);
	cudaFree(d_cos_table);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_nlines);
	cudaFree(d_image_in);
	cudaFree(d_image_out);
    cudaFree(d_imEdge);
    cudaFree(d_NR);
    cudaFree(d_G);
    cudaFree(d_phi);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
    cudaFree(d_pedge);
    cudaFree(d_accum);
}
