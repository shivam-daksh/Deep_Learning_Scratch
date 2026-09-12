/*
clang++ -std=c++17 benchmark.cpp -framework OpenCL -o benchmark
*/
#define CL_SILENCE_DEPRECATION 
#include <OpenCL/opencl.h>     
#include <iostream>
#include <vector>
#include <string>
#include <chrono> // For high-resolution timing

// Simple element-wise multiplication kernel to keep vectors busy
const char* kernelSource = R"(
__kernel void vector_stress(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    float a = A[id];
    float b = B[id];
    float res = 0.0f;
    for(int i = 0; i < 50; i++) {
        res += (a * b) + 0.01f;
    }
    C[id] = res;
}
)";

int main() {
    // 64 Million elements (~256 Megabytes per array)
    const int DATA_SIZE = 64 * 1024 * 1024; 
    // Reduced loop passes so the CPU phase doesn't freeze your terminal for minutes
    const int ITERATIONS = 20; 

    std::cout << "Allocating memory buffers (Total ~768MB Host RAM)..." << std::endl;
    std::vector<float> h_A(DATA_SIZE, 1.05f);
    std::vector<float> h_B(DATA_SIZE, 0.95f);
    std::vector<float> h_C_cpu(DATA_SIZE, 0.0f);
    std::vector<float> h_C_gpu(DATA_SIZE, 0.0f);

    // ==========================================
    // PHASE 1: CPU RUN
    // ==========================================
    std::cout << "\n[1/2] Starting CPU Benchmark (" << ITERATIONS << " passes)..." << std::endl;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < ITERATIONS; step++) {
        // Explicitly force sequential CPU execution mirroring the OpenCL kernel logic
        #pragma omp parallel for // Optional: OpenMP could parallelize this, but let's test raw sequential core performance first
        for (int id = 0; id < DATA_SIZE; id++) {
            float a = h_A[id];
            float b = h_B[id];
            float res = 0.0f;
            for (int i = 0; i < 50; i++) {
                res += (a * b) + 0.01f;
            }
            h_C_cpu[id] = res;
        }
        std::cout << "  -> CPU Completed pass " << step + 1 << "/" << ITERATIONS << "..." << std::endl;
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Phase Complete. Sample element C_CPU[0]: " << h_C_cpu[0] << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " seconds." << std::endl;

    // ==========================================
    // PHASE 2: GPU (OpenCL) RUN
    // ==========================================
    std::cout << "\n[2/2] Initializing OpenCL environment..." << std::endl;
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);

    cl_device_id devices[4];
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 4, devices, &num_devices);

    cl_device_id selected_device = devices[0];
    
    // Explicitly search for the dedicated AMD chip
    std::cout << "--- Available GPUs discovered ---\n";
    for(cl_uint i = 0; i < num_devices; i++) {
        char name[128];
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        std::cout << "[" << i << "] " << name << "\n";
        
        std::string devName(name);
        if (devName.find("AMD") != std::string::npos || devName.find("Radeon") != std::string::npos) {
            selected_device = devices[i];
        }
    }

    char finalName[128];
    clGetDeviceInfo(selected_device, CL_DEVICE_NAME, sizeof(finalName), finalName, NULL);
    std::cout << "\nTargeting GPU for Benchmark: " << finalName << "\n";

    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &selected_device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, selected_device, 0, &err);

    std::cout << "Transferring buffers to Video VRAM..." << std::endl;
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, h_A.data(), &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, h_B.data(), &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 1, &selected_device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vector_stress", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);

    size_t globalSize = DATA_SIZE;
    
    std::cout << "Starting GPU execution loop (" << ITERATIONS << " passes)... WATCH ACTIVITY MONITOR!" << std::endl;
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    for(int step = 0; step < ITERATIONS; step++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        clFlush(queue); 
    }
    clFinish(queue); // Wait completely for GPU to sit idle
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;

    std::cout << "GPU Phase Complete. Fetching back sample data..." << std::endl;
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * DATA_SIZE, h_C_gpu.data(), 0, NULL, NULL);
    std::cout << "Sample element C_GPU[0]: " << h_C_gpu[0] << std::endl;
    std::cout << "GPU Time: " << gpu_duration.count() << " seconds." << std::endl;

    // ==========================================
    // PHASE 3: COMPARISON SUMMARY
    // ==========================================
    std::cout << "\n============================================\n";
    std::cout << "             BENCHMARK SUMMARY              \n";
    std::cout << "============================================\n";
    std::cout << "Processed Data: " << ITERATIONS << " passes over " << (DATA_SIZE / 1000000.0) << "M elements.\n";
    std::cout << "CPU Runtime   : " << cpu_duration.count() << " seconds\n";
    std::cout << "GPU Runtime   : " << gpu_duration.count() << " seconds\n";
    
    double speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "--------------------------------------------\n";
    std::cout << "RESULT: The GPU was " << speedup << "x FASTER than the CPU!\n";
    std::cout << "============================================\n";

    // Clean up OpenCL
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}
