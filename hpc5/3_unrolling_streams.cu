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
#define BLOCK_SIZE 256 // Fixed block size for unrolling

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

/* * GPU Kernel: Calculate Forces 
 * OPTIMIZATION 1: Removed multi-system logic. This kernel handles ONE system.
 * OPTIMIZATION 2: Loop Unrolling inside the tile processing.
 */
__global__ void bodyForceKernel(Body *p, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 myPos;

    // Prefetch my position
    if (i < n) {
        myPos.x = p[i].x;
        myPos.y = p[i].y;
        myPos.z = p[i].z;
    }

    // Allocate Shared Memory
    __shared__ float3 tile[BLOCK_SIZE];

    // Loop over all tiles in this system
    // (n + BLOCK_SIZE - 1) / BLOCK_SIZE calculation effectively
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tileIdx = 0; tileIdx < num_tiles; tileIdx++) {
        // 1. Collaborative loading
        int load_idx = tileIdx * blockDim.x + threadIdx.x;
        
        if (load_idx < n) {
            tile[threadIdx.x].x = p[load_idx].x;
            tile[threadIdx.x].y = p[load_idx].y;
            tile[threadIdx.x].z = p[load_idx].z;
        } else {
            tile[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // 2. Compute forces
        // OPTIMIZATION: #pragma unroll helps compiler generate parallel instructions
        // We limit unroll factor to avoid excessive register pressure if BLOCK_SIZE is large
        #pragma unroll 32
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

        __syncthreads();
    }

    if (i < n) {
        p[i].vx += dt * acc.x;
        p[i].vy += dt * acc.y;
        p[i].vz += dt * acc.z;
    }
}

/* GPU Kernel: Integrate positions */
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

    // --- 1. CPU VERSION ---
    printf("Running sequential CPU simulation for %d systems...\n", num_systems);
    
    StartTimer(); 
    
    for (int iter = 1; iter <= nIters; iter++) {
        #pragma omp parallel for private(system_ptr) schedule(static)
        for (int sys = 0; sys < num_systems; sys++) {
            system_ptr = &data[sys * bodies_per_system];
            bodyForce(system_ptr, dt, bodies_per_system);
            integrate(system_ptr, dt, bodies_per_system);
        }
    }
    
    totalTime = GetTimer() / 1000.0; 

    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;

    printf("CPU Total Time: %.3f seconds\n", totalTime);
    printf("CPU Final position of System 0, Body 0: %.4f, %.4f, %.4f\n", data[0].x, data[0].y, data[0].z);

    // --- 2. CUDA VERSION (Optimized with Streams) ---
    printf("\nRunning CUDA simulation with Streams and Unrolling...\n");

    int blockSize = BLOCK_SIZE; 
    // Grid size is now calculated PER SYSTEM
    int gridSize = (bodies_per_system + blockSize - 1) / blockSize;
    
    // OPTIMIZATION: Create Streams
    cudaStream_t *streams = (cudaStream_t *)malloc(num_systems * sizeof(cudaStream_t));
    for (int i = 0; i < num_systems; i++) {
        cudaStreamCreate(&streams[i]);
    }

    GpuTimer timer; 
    timer.Start();

    for (int iter = 1; iter <= nIters; iter++) {
        // Loop over systems and launch independent kernels into separate streams
        // This allows the GPU scheduler to overlap execution of these small grids
        for (int sys = 0; sys < num_systems; sys++) {
            // Calculate offset for this system
            Body *sys_d_data = d_data + (sys * bodies_per_system);

            // Launch Force Kernel in Stream[sys]
            bodyForceKernel<<<gridSize, blockSize, 0, streams[sys]>>>(sys_d_data, dt, bodies_per_system);
            
            // Launch Integrate Kernel in Stream[sys] (enforcing Force -> Integrate dependency within stream)
            integrateKernel<<<gridSize, blockSize, 0, streams[sys]>>>(sys_d_data, dt, bodies_per_system);
        }
    }
    
    // Wait for all streams to finish
    cudaDeviceSynchronize();

    timer.Stop();
    
    // Cleanup Streams
    for (int i = 0; i < num_systems; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    double gpuTime = timer.Elapsed() / 1000.0;

    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("GPU Total Time: %.3f seconds\n", gpuTime);
    printf("GPU Average Throughput: %0.3f Billion Interactions / second\n", 1e-9 * total_interactions / gpuTime);
    printf("GPU Final position of System 0, Body 0: %.4f, %.4f, %.4f\n", data[0].x, data[0].y, data[0].z);

    cudaFree(d_data);
    free(data);
    return 0;
}
