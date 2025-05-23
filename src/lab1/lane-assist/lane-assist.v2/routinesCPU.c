#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "routinesCPU.h"
#include "png_io.h"

#define DEG2RAD 0.017453f

uint8_t *image_RGB2BW(uint8_t *image_in, int height, int width)
{
	int i, j;
	uint8_t *imageBW = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float R, B, G;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			R = (float)(image_in[3 * (i * width + j)    ]);
			G = (float)(image_in[3 * (i * width + j) + 1]);
			B = (float)(image_in[3 * (i * width + j) + 2]);

			imageBW[i * width + j] = (uint8_t)(0.2989 * R + 0.5870 * G + 0.1140 * B);
		}

	return imageBW;
}

void init_cos_sin_table(float *sin_table, float *cos_table, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		sin_table[i] = sinf(i * DEG2RAD);
		cos_table[i] = cosf(i * DEG2RAD);
	}
}

void canny(uint8_t *im, uint8_t *image_out,
		   float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
		   float level,
		   int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
		{
			// Noise reduction
			NR[i * width + j] =
				(2.0 * im[(i - 2) * width + (j - 2)] + 4.0 * im[(i - 2) * width + (j - 1)] + 5.0 * im[(i - 2) * width + (j)] + 4.0 * im[(i - 2) * width + (j + 1)] + 2.0 * im[(i - 2) * width + (j + 2)] + 4.0 * im[(i - 1) * width + (j - 2)] + 9.0 * im[(i - 1) * width + (j - 1)] + 12.0 * im[(i - 1) * width + (j)] + 9.0 * im[(i - 1) * width + (j + 1)] + 4.0 * im[(i - 1) * width + (j + 2)] + 5.0 * im[(i)*width + (j - 2)] + 12.0 * im[(i)*width + (j - 1)] + 15.0 * im[(i)*width + (j)] + 12.0 * im[(i)*width + (j + 1)] + 5.0 * im[(i)*width + (j + 2)] + 4.0 * im[(i + 1) * width + (j - 2)] + 9.0 * im[(i + 1) * width + (j - 1)] + 12.0 * im[(i + 1) * width + (j)] + 9.0 * im[(i + 1) * width + (j + 1)] + 4.0 * im[(i + 1) * width + (j + 2)] + 2.0 * im[(i + 2) * width + (j - 2)] + 4.0 * im[(i + 2) * width + (j - 1)] + 5.0 * im[(i + 2) * width + (j)] + 4.0 * im[(i + 2) * width + (j + 1)] + 2.0 * im[(i + 2) * width + (j + 2)]) / 159.0;
		}

	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
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

	// Edge
	for (i = 3; i < height - 3; i++)
		for (j = 3; j < width - 3; j++)
		{
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

	// Hysteresis Thresholding
	lowthres = level / 2;
	hithres = 2 * (level);

	for (i = 3; i < height - 3; i++)
		for (j = 3; j < width - 3; j++)
		{
			image_out[i * width + j] = 0;
			if (G[i * width + j] > hithres && pedge[i * width + j])
				image_out[i * width + j] = 255;
			else if (pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
				// check neighbours 3x3
				for (ii = -1; ii <= 1; ii++)
					for (jj = -1; jj <= 1; jj++)
						if (G[(i + ii) * width + j + jj] > hithres)
							image_out[i * width + j] = 255;
		}
}

void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
	float *sin_table, float *cos_table)
{
	int i, j;

	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);

	for (i = 0; i < accu_width * accu_height; i++)
		accumulators[i] = 0;

	float center_x = width / 2.0;
	float center_y = height / 2.0;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			if (im[(i * width) + j] > 250) // Pixel is edge
			{
				for (int theta = 0; theta < 180; theta++)
				{
					float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[(int)((round(rho + hough_h) * 180.0)) + theta]++;
				}
			}
		}
}

void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height,
			  float *sin_table, float *cos_table,
			  int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for (rho = 0; rho < accu_height; rho++)
		for (theta = 0; theta < accu_width; theta++)
		{
			if (accumulators[(rho * accu_width) + theta] >= threshold)
			{
				// Is this point a local maxima (9x9)
				max = accumulators[(rho * accu_width) + theta];
				for (ii = -4; ii <= 4; ii++)
				{
					for (jj = -4; jj <= 4; jj++)
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
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
}

void draw_lines(uint8_t *imgtmp, int width, int height, int *x1, int *y1, int *x2, int *y2, int nlines)
{
	int x, y, wl, l;
	int width_line = 9;

	for (l = 0; l < nlines; l++)
		for (wl = -(width_line >> 1); wl <= (width_line >> 1); wl++)
			for (x = x1[l]; x < x2[l]; x++)
			{
				y = (float)(y2[l] - y1[l]) / (x2[l] - x1[l]) * (x - x1[l]) + y1[l]; // Line eq. known two points
				if (x + wl > 0 && x + wl < width && y > 0 && y < height)
				{
					imgtmp[3 * ((y)*width + x + wl)	   ] = 255;
					imgtmp[3 * ((y)*width + x + wl) + 1] = 0;
					imgtmp[3 * ((y)*width + x + wl) + 2] = 0;
				}
			}
}

void lane_assist_CPU(uint8_t *imgtmp, int height, int width,
					 uint8_t *imEdge, int accu_height, int accu_width,
					 int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	float sin_table[180], cos_table[180];
	// Initialize sin and cos tables
	init_cos_sin_table(sin_table, cos_table, 180);

	// Convert image RGB to BW
	uint8_t *im = image_RGB2BW(imgtmp, height, width);

	// Create temporal buffers
	float *NR   	= (float *)  malloc(sizeof(float) 	* width * height);
	float *G   		= (float *)  malloc(sizeof(float) 	* width * height);
	float *phi   	= (float *)  malloc(sizeof(float) 	* width * height);
	float *Gx   	= (float *)  malloc(sizeof(float) 	* width * height);
	float *Gy 		= (float *)  malloc(sizeof(float) 	* width * height);
	uint8_t *pedge  = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	uint32_t *accum = (uint32_t*)malloc(sizeof(uint32_t)*accu_width*accu_height);

	// Canny (edge detection)
	canny(im, imEdge,
		  NR, G, phi, Gx, Gy, pedge,
		  1000.0f, // level
		  height, width);

	// Hough transform (line detection)
	houghtransform(imEdge, width, height,
				   accum, accu_width, accu_height,
				   sin_table, cos_table);

	int threshold;
	if (width > height) threshold = width / 6;
	else				threshold = height / 6;

	getlines(threshold, accum,
			 accu_width, accu_height,
			 width, height,
			 sin_table, cos_table,
			 x1, y1, x2, y2, nlines);

	draw_lines(imgtmp, width, height, x1, y1, x2, y2, *nlines);
}
