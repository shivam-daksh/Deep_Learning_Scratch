/*
clang++ -std=c++17 gputest.cpp -framework OpenCL -o gputest
*/
#define CL_SILENCE_DEPRECATION 
#include <OpenCL/opencl.h>     
#include <iostream>
#include <vector>
#include <string>

// Simple element-wise multiplication kernel to keep vectors busy
const char* kernelSource = R"(
__kernel void vector_stress(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    // Artificially inflate work per thread to make the GPU work harder
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
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);

    // Get ALL available GPU devices (Intel and AMD will both show up here)
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
        
        // If "AMD" or "Radeon" is in the name, prioritize it over Intel!
        std::string devName(name);
        if (devName.find("AMD") != std::string::npos || devName.find("Radeon") != std::string::npos) {
            selected_device = devices[i];
        }
    }

    char finalName[128];
    clGetDeviceInfo(selected_device, CL_DEVICE_NAME, sizeof(finalName), finalName, NULL);
    std::cout << "\nTargeting GPU for Stress Test: " << finalName << "\n\n";

    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &selected_device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, selected_device, 0, &err);

    // 64 Million elements (~256 Megabytes per array)
    const int DATA_SIZE = 64 * 1024 * 1024; 
    std::cout << "Allocating memory buffers (Total ~768MB Host RAM)..." << std::endl;
    std::vector<float> h_A(DATA_SIZE, 1.05f);
    std::vector<float> h_B(DATA_SIZE, 0.95f);
    std::vector<float> h_C(DATA_SIZE, 0.0f);

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
    const int ITERATIONS = 1500; // Run it repeatedly to give you time to look at Activity Monitor
    
    std::cout << "Starting heavy execution loop (" << ITERATIONS << " passes)... WATCH ACTIVITY MONITOR NOW!" << std::endl;
    
    for(int step = 0; step < ITERATIONS; step++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        // Flush commands to the GPU immediately to prevent bundling delay
        clFlush(queue); 
        if(step % 30 == 0 && step > 0) {
            std::cout << "  -> Completed " << step << " compute passes..." << std::endl;
        }
    }
    
    clFinish(queue); // Block until the final queue step is completely finished
    std::cout << "GPU computation complete! Fetching back sample data..." << std::endl;

    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * DATA_SIZE, h_C.data(), 0, NULL, NULL);
    std::cout << "Sample element C[0]: " << h_C[0] << std::endl;

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}
