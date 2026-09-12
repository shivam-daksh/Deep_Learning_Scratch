/*
clang++ -std=c++17 test.cpp -framework OpenCL -o opencl_test
*/
#define CL_SILENCE_DEPRECATION // Silences macOS deprecation warnings
#include <OpenCL/opencl.h>     // Note the OpenCL/ prefix for macOS
#include <iostream>
#include <vector>

// Simple GPU kernel source code written in OpenCL C (C99-based)
const char* kernelSource = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}
)";

int main() {
    // 1. Discover platforms and select the AMD/GPU device
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms, num_devices;

    if (clGetPlatformIDs(1, &platform, &num_platforms) != CL_SUCCESS) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        return 1;
    }

    // Explicitly look for a GPU device
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cout << "GPU not found or failing. Falling back to CPU for testing...\n";
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
        if (err != CL_SUCCESS) {
            std::cerr << "No OpenCL compute devices found at all.\n";
            return 1;
        }
    }

    // Print device name
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    std::cout << "Using Device: " << deviceName << "\n\n";

    // 2. Create an OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    // macOS supports clCreateCommandQueue (deprecated but works perfectly on Intel Macs)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // 3. Prepare Dummy Data
    const int DATA_SIZE = 1024;
    std::vector<float> h_A(DATA_SIZE, 1.0f); // Host input A (all 1.0)
    std::vector<float> h_B(DATA_SIZE, 2.0f); // Host input B (all 2.0)
    std::vector<float> h_C(DATA_SIZE, 0.0f); // Host output C

    // 4. Create GPU buffers and copy host data over
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, h_A.data(), &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, h_B.data(), &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, &err);

    // 5. Create, build, and compile the OpenCL code at runtime
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);

    // Set the arguments for the kernel function
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);

    // 6. Execute the kernel on the GPU
    size_t globalSize = DATA_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Wait for the GPU to finish execution
    clFinish(queue);

    // 7. Copy the results from GPU memory back to CPU RAM
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * DATA_SIZE, h_C.data(), 0, NULL, NULL);

    // 8. Verify the math (1.0 + 2.0 should equal 3.0)
    bool success = true;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS! OpenCL executed perfectly on your GPU.\n";
        std::cout << "Sample element C[0]: " << h_C[0] << " (Expected: 3)\n";
    } else {
        std::cerr << "FAILURE: The calculations returned unexpected data.\n";
    }

    // Clean up OpenCL objects
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}
