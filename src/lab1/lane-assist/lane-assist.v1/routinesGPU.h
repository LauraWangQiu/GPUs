#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

/**
 * @param[in] im input image
 * @param[in] height image height
 * @param[in] width image width
 * @param[inout] imEdge edge image
 * @param[in] accu_height accumulator height
 * @param[in] accu_width accumulator width
 * @param[inout] x1 starting x coordinate of the line
 * @param[inout] y1 starting y coordinate of the line
 * @param[inout] x2 ending x coordinate of the line
 * @param[inout] y2 ending y coordinate of the line
 * @param[inout] nlines number of lines
 */
void lane_assist_GPU(uint8_t *imgtmp, int height, int width,
	uint8_t *imEdge, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines);

#endif

