#include "kernel.cuh"

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    particles[idx].posX += particles[idx].velX * deltaTime;
    particles[idx].posY += particles[idx].velY * deltaTime;

    particles[idx].timeLeft -= deltaTime;

    particles[idx].velY += gravityForce * deltaTime;
}

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, float gravityForce) {
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    applyGravityForce<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, gravityForce);

    cudaMemcpy(particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}
