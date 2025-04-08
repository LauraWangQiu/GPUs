#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.h"

__global__ void updateParticles(Particle* particles, int numParticles, float deltaTime);

void updateParticles_kernel(Particle* particles, int numParticles, float deltaTime);
