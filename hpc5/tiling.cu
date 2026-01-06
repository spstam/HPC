#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

// Include the separate timer headers
#include "timer.h"
#include "gpu_timer.h"

#define SOFTENING 0.01f
#define BLOCK_SIZE 256 // Optimization: Defined block size for shared memory

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/* CPU: Calculate forces */
void bodyForce(Body * p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
        Fx = 0.0f; Fy = 0.0f; Fz = 0.0f;

        for (j = 0; j < n; j++) {
            dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/* CPU: Integrate positions */
void integrate(Body * p, float dt, int n) {
    int i;
    for (i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/* * STEP 1 OPTIMIZATION: Shared Memory Tiling 
 * This reduces global memory accesses by loading a "tile" of bodies
 * into shared memory, which all threads in the block can then read quickly.
 */
__global__ void bodyForceKernel(Body *p, float dt, int n, int n_per_system) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float3 acc = {0.0f, 0.0f, 0.0f}; // Force accumulator
    float3 myPos;

    // Cache "my" position in registers to avoid reading from global memory repeatedly
    if (i < n) {
        myPos.x = p[i].x;
        myPos.y = p[i].y;
        myPos.z = p[i].z;
    }

    // Determine which system (galaxy) this body belongs to
    // We only calculate forces from bodies within the SAME system.
    int system_id = i / n_per_system;
    int system_start = system_id * n_per_system;
    
    // Calculate how many tiles we need to cover the bodies in this system
    // Assumption: n_per_system is a multiple of blockDim.x
    int num_tiles = n_per_system / blockDim.x; 

    // Allocate Shared Memory
    __shared__ float3 tile[BLOCK_SIZE];

    for (int tileIdx = 0; tileIdx < num_tiles; tileIdx++) {
        // 1. Collaborative loading: Each thread loads one body from the current tile into shared memory
        int load_idx = system_start + tileIdx * blockDim.x + threadIdx.x;
        
        if (load_idx < n) {
            tile[threadIdx.x].x = p[load_idx].x;
            tile[threadIdx.x].y = p[load_idx].y;
            tile[threadIdx.x].z = p[load_idx].z;
        } else {
            // Padding for safety, though 4096 is a multiple of 256
            tile[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        }

        // 2. Synchronize to ensure the tile is fully loaded
        __syncthreads();

        // 3. Compute forces using the tile in shared memory
        // This loop reads from FAST shared memory, not SLOW global memory
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float dx = tile[j].x - myPos.x;
            float dy = tile[j].y - myPos.y;
            float dz = tile[j].z - myPos.z;
            
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            acc.x += dx * invDist3;
            acc.y += dy * invDist3;
            acc.z += dz * invDist3;
        }

        // 4. Synchronize before loading the next tile (so we don't overwrite data others are using)
        __syncthreads();
    }

    // Write results back to global memory
    if (i < n) {
        p[i].vx += dt * acc.x;
        p[i].vy += dt * acc.y;
        p[i].vz += dt * acc.z;
    }
}

/* GPU Kernel: Integrate positions (Standard) */
__global__ void integrateKernel(Body *p, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char *argv[]) {
    int num_systems = 16;
    int bodies_per_system = 4096;
    int nIters = 10; 
    const float dt = 0.01f;
    
    FILE *fp;
    int total_bodies;
    size_t bytes;
    Body *data, *system_ptr;
    float *buf;
    double totalTime, interactions_per_system, total_interactions;

    /* Dataset setup */
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    total_bodies = num_systems * bodies_per_system;
    bytes = total_bodies * sizeof(Body);
    data = (Body *) malloc(bytes);
    
    if (fp) {
        fread(data, sizeof(Body), total_bodies, fp);
        fclose(fp);
    } else {
        buf = (float *) data;
        for (int i = 0; i < 6 * total_bodies; i++) {
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    Body *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);

    // --- 1. CPU VERSION (using timer.h) ---
    printf("Running sequential CPU simulation for %d systems...\n", num_systems);
    
    StartTimer(); // From timer.h
    
    for (int iter = 1; iter <= nIters; iter++) {
        #pragma omp parallel for private(system_ptr) schedule(static)
        for (int sys = 0; sys < num_systems; sys++) {
            system_ptr = &data[sys * bodies_per_system];
            bodyForce(system_ptr, dt, bodies_per_system);
            integrate(system_ptr, dt, bodies_per_system);
        }
    }
    
    totalTime = GetTimer() / 1000.0; // From timer.h (returns ms)

    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;

    printf("CPU Total Time: %.3f seconds\n", totalTime);
    printf("CPU Final position of System 0, Body 0: %.4f, %.4f, %.4f\n", data[0].x, data[0].y, data[0].z);

    // --- 2. CUDA VERSION (using gpu_timer.h) ---
    printf("\nRunning CUDA simulation with Shared Memory Tiling...\n");

    // Optimization: Block Size must match the shared memory tile size (BLOCK_SIZE)
    int blockSize = BLOCK_SIZE; 
    int gridSize = (total_bodies + blockSize - 1) / blockSize;
    
    GpuTimer timer; // From gpu_timer.h
    timer.Start();

    for (int iter = 1; iter <= nIters; iter++) {
        bodyForceKernel<<<gridSize, blockSize>>>(d_data, dt, total_bodies, bodies_per_system);
        integrateKernel<<<gridSize, blockSize>>>(d_data, dt, total_bodies);
    }
    
    timer.Stop();
    
    // GpuTimer.Elapsed() returns milliseconds, convert to seconds
    double gpuTime = timer.Elapsed() / 1000.0;

    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("GPU Total Time: %.3f seconds\n", gpuTime);
    printf("GPU Average Throughput: %0.3f Billion Interactions / second\n", 1e-9 * total_interactions / gpuTime);
    printf("GPU Final position of System 0, Body 0: %.4f, %.4f, %.4f\n", data[0].x, data[0].y, data[0].z);

    cudaFree(d_data);
    free(data);
    return 0;
}
