#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16

#define TIMEVAL_TO_SECONDS(timeval) timeval.tv_sec + timeval.tv_usec * 1E-6
#define TIME_DIFF_SECONDS(timeval1, timeval2) TIMEVAL_TO_SECONDS(timeval1) - TIMEVAL_TO_SECONDS(timeval2)
#define FLOPS(m, n, k) 2 * m * n * k

#define VERSION_3

#if defined VERSION_1 || defined VERSION_2
// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);
#elif defined VERSION_3
// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);
#endif

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
	struct timeval init, end;
	gettimeofday(&init,NULL);
	cudaMalloc((void**)&Ad, sizeA);
	cudaMemcpy(Ad, A, sizeA, cudaMemcpyHostToDevice);
	gettimeofday(&end,NULL);
	double tt1 = TIME_DIFF_SECONDS(end, init);

	float* Bd;
	int sizeB = /*hB*/wA * wB * sizeof(float);
	gettimeofday(&init,NULL);
	cudaMalloc((void**)&Bd, sizeB);
	cudaMemcpy(Bd, B, sizeB, cudaMemcpyHostToDevice);
	gettimeofday(&end,NULL);
	double tt2 = TIME_DIFF_SECONDS(end, init);

	// Allocate C on the device
	float* Cd;
	int sizeC = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, sizeC);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(ceil((float)wB / dimBlock.x), ceil((float)hA / dimBlock.y));
	gettimeofday(&init,NULL);
	// Launch the device computation
#ifdef VERSION_1
	// Version 1
	Muld<<<1, 1>>>(Ad, Bd, wA, wB, hA, Cd);
#elif defined VERSION_2
	// Version 2_ using thread and block identificators
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, hA, Cd);
#elif defined VERSION_3
	// Version 3: using shared memory
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, hA, wA, wB, Cd);
#endif
	cudaDeviceSynchronize(); // Bloquear
	gettimeofday(&end,NULL);
	double tkernel = TIME_DIFF_SECONDS(end, init);

	gettimeofday(&init,NULL);
	// Read C from the device
	cudaMemcpy(C, Cd, sizeC, cudaMemcpyDeviceToHost);
	gettimeofday(&end,NULL);
	double tt3 = TIME_DIFF_SECONDS(end, init);

	printf("GPU:\n");
	printf("Time A (malloc and memcpy): %lf s\n", tt1);
	printf("Time B (malloc and memcpy): %lf s\n", tt2);
	printf("Time Kernel: %lf s\n", tkernel);
	printf("Time C (cudaMemcpyDeviceToHost): %lf s\n", tt3);
	printf("Bandwidth of A: %lf KB/s\n", sizeA * 1E-3/tt1);
	printf("Bandwidth of B: %lf KB/s\n", sizeB * 1E-3/tt2);
	printf("Performance Kernel: %lf GFLOPS/s\n", FLOPS(hA, wB, wA) * 1E-9/tkernel);
	printf("Bandwidth of C: %lf KB/s\n", sizeC * 1E-3/tt3);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}

// Funcion asincrona
#if defined VERSION_1 || defined VERSION_2
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
	// Version 2: using thread and block identificators
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
#elif defined VERSION_3
// Device multiplication function called by Mul()
// Compute C = A * B
// hA is the height of A
// wA is the width of A and the height of B
// wB is the width of B
__global__ void Muld(float* A, float* B, int hA, int wA, int wB, float* C)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = BLOCK_SIZE * wA * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

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
		// As[ty][tx] = A[a + ty * wA + tx];
		// Bs[ty][tx] = B[b + ty * wB + tx];
		// If is not out of bounds, load the element
		// If is out of bounds, load 0
		if (a + ty * wA + tx < wA * hA)
            As[ty][tx] = A[a + ty * wA + tx];
        else
            As[ty][tx] = 0.0;

        if (b + ty * wB + tx < wB * wA /*hB*/)
            Bs[ty][tx] = B[b + ty * wB + tx];
        else
            Bs[ty][tx] = 0.0;

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	// C[(ty + by * blockDim.y) * wB + (tx + bx * blockDim.x)] = Csub;
	// If is not out of bounds, write the result
	int row = ty + by * blockDim.y;
	int col = tx + bx * blockDim.x;
	if (row < hA && col < wB)
		C[row * wB + col] = Csub;
}
#endif
