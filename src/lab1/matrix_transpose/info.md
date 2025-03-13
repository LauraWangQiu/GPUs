# Matrix Transpose

## To Do

- Crear diferentes versiones de transpose.cu en diferentes directorios CUDA.v1, CUDA.v2, etc.
- Establecer el numero de hilos relacionado al numero de cores

## Notes

- OpenMP, paralelizar código con directivas #pragma omp parallel for
- Hay que pasarle la flag de -fopenmp en gcc para que paralelice
- Se puede especificar el numero de hilos de OpenMP exportando OMP_NUM_THREADS, OMP_NUM_THREADS=1 ./transpose
- Hyperthreading, hilos por core físico
- Algunos códigos no pueden explotar el hardware porque por ejemplo en operaciones aritmetico-logicas no se puede superar el numero de modulos (ALUs) (compute bound)

## Profiler

```bash
# Perfilar
nsys nvprof ./transpose
# Visualización de la traza
ncu-ui report.nsys-rep
# Recolectar profiling
ncu -o profile --set full ./transpose
```

## CUDA.v1

```c
__global__ void transpose_device(float *in, float *out, int rows, int cols) 
{ 
    int i, j; 
    i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i<rows)
        for ( j=0; j<cols; j++) 
            out [ i * rows + j ] = in [ j * cols + i ]; 
}
```

La versión 1 para calcular la matriz traspuesta es ineficiente porque utiliza una cuadrícula unidimensional de subprocesos y un bucle interno para manejar las columnas de la matriz. Este enfoque no aprovecha al máximo las capacidades de procesamiento en paralelo de la GPU.

## CUDA.v2

```c
#define NTHREADS2D 16 // 16x16=256 threads (16 threads per block)

__global__ void transpose_device(float *in, float *out, int rows, int cols) 
{ 
    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x; 
    j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < rows && j < cols)
        out[i * rows + j] = in[j * cols + i]; 
}
```

> IMPORTANTE: un warp es un conjunto de 32 hilos que se ejecutan simultáneamente en un multiprocesador de streaming (SM).
> Leer más en: [CUDA](https://es.wikipedia.org/wiki/CUDA)

Con `NTHREADS2D` como 16 da como resultado bloques de 256 hilos (16x16). Este tamaño es eficiente porque es un múltiplo de 32 y se alinea con la estructura de los warps en CUDA. Sin embargo, si NTHREADS2D se establece en un valor mayor que 32, el número total de hilos por bloque excedería 1024 (por ejemplo, 33x33 = 1089), que es el límite máximo de hilos por bloque en muchas arquitecturas CUDA.

Leer más en: [Thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming))

## CUDA.v3

```c
#define NTHREADS2D 16 // 16x16=256 threads (16 threads per block)
#define TILE_DIM 16

__global__ void transpose_device(float *in, float *out, int rows, int cols) 
{ 
    int i, j; 
	__shared__ float tile [ TILE_DIM ] [ TILE_DIM ]; 

	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 

	if (i<rows && j<cols) {
		tile[threadIdx.y] [threadIdx.x] = in[ j * cols + i ];
		__syncthreads(); 
		i = blockIdx.x * TILE_DIM + threadIdx.y;
		j = blockIdx.y * TILE_DIM + threadIdx.x; 
		out[ i * rows + j ] = tile[threadIdx.x] [threadIdx.y];
	}
}
```

Usamos memoria compartida para almacenar los elementos de la matriz de entrada antes de escribirlos en la matriz de salida. La memoria compartida es mucho más rápida que la memoria global, por lo que este enfoque mejora significativamente el rendimiento.

## CUDA.v4

```c
#define NTHREADS2D 16 // 16x16=256 threads (16 threads per block)
#define TILE_DIM 16

__global__ void transpose_device(float *in, float *out, int rows, int cols) 
{ 
    int i, j; 
	__shared__ float tile [ TILE_DIM ] [ TILE_DIM+1 ]; 

	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y; 

	if (i<rows && j<cols) {
		tile[threadIdx.y] [threadIdx.x] = in[ j * cols + i ];
		__syncthreads(); 
		i = blockIdx.x * TILE_DIM + threadIdx.y;
		j = blockIdx.y * TILE_DIM + threadIdx.x; 
		out[ i * rows + j ] = tile[threadIdx.x] [threadIdx.y];
	}
}
```

Añadimos padding para evitar el bank conflict en la memoria compartida. El bank conflict ocurre cuando dos o más hilos intentan acceder a la misma ubicación de memoria compartida. En este caso, los accesos a la memoria compartida se dividen en bancos de 32 bits, y si dos o más hilos intentan acceder a la misma ubicación de memoria compartida, se produce un conflicto de bancos.