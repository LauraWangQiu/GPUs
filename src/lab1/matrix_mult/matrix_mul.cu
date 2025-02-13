#include <stdio.h>
#include <sys/time.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16 

// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);

	struct timeval init, end;
	gettimeofday(&init,NULL);

	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);

	gettimeofday(&end,NULL);

	printf("Time A: %ld ms\n", end.tv_usec - init.tv_usec);

	float* Bd;
	size = wA * wB * sizeof(float);

	gettimeofday(&init,NULL);

	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	gettimeofday(&end,NULL);

	printf("Time B: %ld ms\n", end.tv_usec - init.tv_usec);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// // Compute the execution configuration assuming
	// // the matrix dimensions are multiples of BLOCK_SIZE
	// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);

	gettimeofday(&init,NULL);

	// // Launch the device computation
	// Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, hA, Cd);
	Muld<<<1, 1>>>(Ad, Bd, wA, wB, hA, Cd); // Funcion asincrona
	cudaDeviceSynchronize(); // Bloquear

	gettimeofday(&end,NULL);

	//suseconds_t bandWidth = 
	printf("Time Muld: %ld ms\n", end.tv_usec - init.tv_usec);

	gettimeofday(&init,NULL);

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	gettimeofday(&end,NULL);

	printf("Time cudaMemcpyDeviceToHost (C): %ld ms\n", end.tv_usec - init.tv_usec);



	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}

__global__ void Muld(float* A, float* B, int wA, int wB, int hA, float* C)
{
	for (int i = 0; i < hA; ++i) {
		for (int j = 0; j < wB; ++j) {
			C[i * wB + j] = 0.0f;
			for (int k = 0; k < wA; ++k) {
				C[i * wB + j] += A[i * wA + k] * B[k * wB + j];
			}
		}
	}
}

#if 0
// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = ...;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = ...;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to
	// compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrix
		As[ty][tx] = A[...];
		Bs[ty][tx] = B[...];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			....

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	...
}
#endif
