TTx1; TTx2; TKernel; TTx3; BWTx1; BWTx2; PERFKernel; BWTx3;

BW = bandwidth (KB/s)
PERFKernel = performance of the kernel (GFLOPS/s)
FLOPS = 2 * m * n * k;

BWTx1; BWTx2; más rapidos que BWTx3

Queremos transformar el kernel para que cada hilo se encargue de cada posicion de la matriz de C

Probar con valores multiplos del BLOCK_SIZE(16)