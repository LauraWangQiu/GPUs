- Crear diferentes versiones de transpose.cu en diferentes directorios CUDA.v1, CUDA.v2, etc.
- OpenMP, paralelizar código con directivas #pragma omp parallel for
- Hay que pasarle la flag de -fopenmp en gcc para que paralelice
- Se puede especificar el numero de hilos de OpenMP exportando OMP_NUM_THREADS, OMP_NUM_THREADS=1 ./transpose
- Establecer el numero de hilos relacionado al numero de cores
- Hyperthreading, hilos por core físico
- Algunos códigos no pueden explotar el hardware porque por ejemplo en operaciones aritmetico-logicas no se puede superar el numero de modulos (ALUs) (compute bound)
