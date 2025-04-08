# nbody

> **Autora:** Yi (Laura) Wang Qiu

Se han implementado funciones que usan SYCL llamadas `get_acceleration_kernel` y `updateParticles_kernel`, y la versión de uso de estructura de arrays (SoA).

Para cambiar entre SoA y AoS, dirigirse a [GSimulation.cpp](./GSimulation.cpp) y cambiar `#define SoA` por `#define AoS` respectivamente.

Para cambiar entre la versión CPU y GPU, dirigirse también a [GSimulation.cpp](./GSimulation.cpp) y dejar definida `#define GPU` para GPU o quitar la línea para CPU.

## Evaluar si el esquema utilizado de memoria AoS es el más adecuado para la explotación de paralelismo en una GPU

El esquema de memoria AoS (Array of Structures) no es el más adecuado para este caso porque se produce strides en la memoria. Esto provoca que el acceso a la memoria no sea coalescente reduciendo el rendimiento de la GPU. Los strides son ocasionados porque por cada partícula se accede a cada uno de sus atributos (posición (x,y,z), velocidad (x,y,z), aceleración (x,y,z) y masa) de forma separada. Esto provoca que los accesos a memoria no sean contiguos, lo que reduce la eficiencia del acceso a memoria en la GPU. En cambio, el esquema SoA (Structure of Arrays) es más adecuado para este caso porque permite acceder a los atributos de las partículas de forma contigua, lo que mejora la coalescencia de los accesos a memoria y, por lo tanto, el rendimiento de la GPU.

## Cálculo de la energía (energy) en updateParticles

Se ha decidido implementar `reductionAtomics1` para calcular la energía en `updateParticles_kernel`.