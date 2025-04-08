#include "kernel.cuh"

__global__ void updateParticles(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    particles[idx].posX += particles[idx].velX * deltaTime;
    particles[idx].posY += particles[idx].velY * deltaTime;

    particles[idx].timeLeft -= deltaTime;

    particles[idx].velY += 100.8f * deltaTime;
}

void updateParticles_kernel(Particle* particles, int numParticles, float deltaTime) {
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime);

    cudaMemcpy(particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}
