#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.h"

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce);

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, float gravityForce = 9.8f);
