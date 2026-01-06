#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "timer.h"
#include "gpu_timer.h"

#define SOFTENING 0.01f
#define BLOCK_SIZE 256
#define COARSENING_FACTOR 4 // Optimization: Each thread processes 4 bodies

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/* CPU Code (Validation) */
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

void integrate(Body * p, float dt, int n) {
    int i;
    for (i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/* * STEP 3 OPTIMIZATION: Thread Coarsening
 * Each thread computes forces for COARSENING_FACTOR (4) bodies.
 * This reuses the shared memory tile 4 times, reducing overhead.
 */
__global__ void bodyForceKernel(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z, 
                                float * __restrict__ vx, float * __restrict__ vy, float * __restrict__ vz, 
                                float dt, int n, int n_per_system) {
    
    // We calculate the start index for this thread's batch of bodies.
    // The block handles (BLOCK_SIZE * COARSENING_FACTOR) bodies total.
    int blockStart = blockIdx.x * blockDim.x * COARSENING_FACTOR;
    int threadStart = blockStart + threadIdx.x;

    // Registers to hold positions and force accumulators for 4 bodies
    float3 myPos[COARSENING_FACTOR];
    float3 acc[COARSENING_FACTOR];

    // Initialize accumulators and load my positions
    // We stride by blockDim.x to maintain coalesced reading for the first load
    #pragma unroll
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        int idx = threadStart + k * blockDim.x;
        acc[k] = make_float3(0.0f, 0.0f, 0.0f);
        if (idx < n) {
            myPos[k] = make_float3(x[idx], y[idx], z[idx]);
        }
    }

    // Identify system limits
    // Note: This assumes n_per_system divides evenly by our coarse block size
    // For 4096 bodies and factor 4, it works perfectly.
    int system_id = threadStart / n_per_system; 
    int system_start = system_id * n_per_system;
    int num_tiles = n_per_system / BLOCK_SIZE;

    __shared__ float3 tile[BLOCK_SIZE];

    for (int tileIdx = 0; tileIdx < num_tiles; tileIdx++) {
        // 1. Collaborative Loading (Same as before)
        // Even though each thread processes 4 bodies, we still only use 
        // 256 threads to load 256 elements into the tile.
        int load_idx = system_start + tileIdx * BLOCK_SIZE + threadIdx.x;
        
        if (load_idx < n) {
            tile[threadIdx.x] = make_float3(x[load_idx], y[load_idx], z[load_idx]);
        } else {
            tile[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // 2. Compute Interactions
        // For every body in the tile...
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float3 sPos = tile[j]; // Cache shared mem value in register
            
            // ...update all 4 of my bodies
            #pragma unroll
            for (int k = 0; k < COARSENING_FACTOR; k++) {
                float dx = sPos.x - myPos[k].x;
                float dy = sPos.y - myPos[k].y;
                float dz = sPos.z - myPos[k].z;
                
                float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                acc[k].x += dx * invDist3;
                acc[k].y += dy * invDist3;
                acc[k].z += dz * invDist3;
            }
        }
        __syncthreads();
    }

    // Write back results
    #pragma unroll
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        int idx = threadStart + k * blockDim.x;
        if (idx < n) {
            vx[idx] += dt * acc[k].x;
            vy[idx] += dt * acc[k].y;
            vz[idx] += dt * acc[k].z;
        }
    }
}

__global__ void integrateKernel(float *x, float *y, float *z, 
                                float *vx, float *vy, float *vz, 
                                float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

int main(const int argc, const char *argv[]) {
    int num_systems = 16;
    int bodies_per_system = 4096;
    int nIters = 10; 
    const float dt = 0.01f;
    
    FILE *fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    int total_bodies = num_systems * bodies_per_system;
    size_t sz = total_bodies * sizeof(float);
    
    // Host Allocation
    Body *data = (Body *) malloc(total_bodies * sizeof(Body));
    if (fp) {
        fread(data, sizeof(Body), total_bodies, fp);
        fclose(fp);
    } else {
        float *buf = (float *) data;
        for (int i = 0; i < 6 * total_bodies; i++) buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    }

    // --- CPU Check ---
    printf("Running sequential CPU simulation for %d systems...\n", num_systems);
    StartTimer();
    for (int iter = 1; iter <= nIters; iter++) {
        #pragma omp parallel for schedule(static)
        for (int sys = 0; sys < num_systems; sys++) {
            Body *ptr = &data[sys * bodies_per_system];
            bodyForce(ptr, dt, bodies_per_system);
            integrate(ptr, dt, bodies_per_system);
        }
    }
    double totalTime = GetTimer() / 1000.0;
    double total_interactions = (double)bodies_per_system*bodies_per_system*num_systems*nIters;
    printf("CPU Final position: %.4f, %.4f, %.4f\n", data[0].x, data[0].y, data[0].z);

    // --- GPU Setup ---
    printf("\nRunning CUDA simulation with Thread Coarsening (Factor %d)...\n", COARSENING_FACTOR);

    float *h_x = (float*)malloc(sz); float *h_y = (float*)malloc(sz); float *h_z = (float*)malloc(sz);
    float *h_vx = (float*)malloc(sz); float *h_vy = (float*)malloc(sz); float *h_vz = (float*)malloc(sz);

    for(int i=0; i<total_bodies; i++) {
        h_x[i] = data[i].x; h_y[i] = data[i].y; h_z[i] = data[i].z;
        h_vx[i] = data[i].vx; h_vy[i] = data[i].vy; h_vz[i] = data[i].vz;
    }

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_x, sz); cudaMalloc(&d_y, sz); cudaMalloc(&d_z, sz);
    cudaMalloc(&d_vx, sz); cudaMalloc(&d_vy, sz); cudaMalloc(&d_vz, sz);

    cudaMemcpy(d_x, h_x, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, sz, cudaMemcpyHostToDevice);

    // IMPORTANT: Grid size is reduced by the Coarsening Factor!
    int blockSize = BLOCK_SIZE; 
    int gridSize = (total_bodies + (blockSize * COARSENING_FACTOR) - 1) / (blockSize * COARSENING_FACTOR);
    
    GpuTimer timer;
    timer.Start();

    for (int iter = 1; iter <= nIters; iter++) {
        bodyForceKernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, total_bodies, bodies_per_system);
        
        // Use standard grid for integrate (it's memory bound, not compute bound, so coarsening matters less)
        int integrateGrid = (total_bodies + blockSize - 1) / blockSize;
        integrateKernel<<<integrateGrid, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, total_bodies);
    }
    
    timer.Stop();
    double gpuTime = timer.Elapsed() / 1000.0;

    cudaMemcpy(h_x, d_x, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, sz, cudaMemcpyDeviceToHost);

    printf("GPU Total Time: %.3f seconds\n", gpuTime);
    printf("GPU Average Throughput: %0.3f Billion Interactions / second\n", 1e-9 * total_interactions / gpuTime);
    printf("GPU Final position: %.4f, %.4f, %.4f\n", h_x[0], h_y[0], h_z[0]);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);
    free(data);
    return 0;
}
