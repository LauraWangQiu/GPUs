#include <stdio.h>
#include <chrono>
#include <math.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16

#define TIMEVAL_TO_SECONDS(timeval) timeval.tv_sec + timeval.tv_usec * 1E-6
#define TIME_DIFF_SECONDS(timeval1, timeval2) TIMEVAL_TO_SECONDS(timeval1) - TIMEVAL_TO_SECONDS(timeval2)
#define FLOPS(m, n, k) 2 * m * n * k

#define VERSION_2

// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B

void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	// Load A and B to the device
	float* Ad;
	int sizeA = hA * wA * sizeof(float);
	auto init = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&Ad, sizeA);
	cudaMemcpy(Ad, A, sizeA, cudaMemcpyHostToDevice);
	auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> tt1 = end - init;

	float* Bd;
	int sizeB = wA * wB * sizeof(float);
	init = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&Bd, sizeB);
	cudaMemcpy(Bd, B, sizeB, cudaMemcpyHostToDevice);
	end = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double> tt2 = end - init;

	// Allocate C on the device
	float* Cd;
	int sizeC = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, sizeC);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(ceil((float)wB / dimBlock.x), ceil((float)hA / dimBlock.y));
	printf("Num %d %d\n", dimGrid.x, dimGrid.y);
	init = std::chrono::high_resolution_clock::now();
#ifdef VERSION_1
	// Version 1
	Muld<<<1, 1>>>(Ad, Bd, wA, wB, hA, Cd);
#elif defined VERSION_2
	// Version 2
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, hA, Cd);
#endif
	cudaDeviceSynchronize(); // Bloquear
	end = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double> tkernel = end - init;

	init = std::chrono::high_resolution_clock::now();
	// Read C from the device
	cudaMemcpy(C, Cd, sizeC, cudaMemcpyDeviceToHost);
	end = std::chrono::high_resolution_clock::now();
	const std::chrono::duration<double> tt3 = end - init;

	printf("Time A (malloc and memcpy): %lf s\n", tt1.count());
	printf("Time B (malloc and memcpy): %lf s\n", tt2.count());
	printf("Time Kernel: %lf s\n", tkernel.count());
	printf("Time C (cudaMemcpyDeviceToHost): %lf s\n", tt3.count());
	printf("Band width of A: %lf KB/s\n", sizeA * 1E-3/tt1.count());
	printf("Band width of B: %lf KB/s\n", sizeB * 1E-3/tt2.count());
	printf("Performance Kernel: %lf GFLOPS/s\n", FLOPS(hA, wB, wA) * 1E-9/tkernel.count());
	printf("Band width of C: %lf KB/s\n", sizeC * 1E-3/tt3.count());

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}

// Funcion asincrona
__global__ void Muld(float* A, float* B, int wA, int wB, int hA, float* C)
{
#ifdef VERSION_1
	// Version 1
	for (int i = 0; i < hA; ++i) {
		for (int j = 0; j < wB; ++j) {
			float accum = 0.0f;
			for (int k = 0; k < wA; ++k) {
				accum += A[i * wA + k] * B[k * wB + j];
			}
			C[i * wB + j] = accum;
		}
	}
#elif defined VERSION_2
	// Version 2
	int i = threadIdx.x + blockIdx.x * blockDim.x; // Componente X
	int j = threadIdx.y + blockIdx.y * blockDim.y; // Componente Y
	if (j < hA && i < wB) {
		float accum = 0.0f;
		for (int k = 0; k < wA; ++k) {
			// j * wA = salto de fila
			// k * wB = salto de columna
			accum += A[j * wA + k] * B[k * wB + i];
		}
		C[j * wB + i] = accum;
	}
#endif
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
