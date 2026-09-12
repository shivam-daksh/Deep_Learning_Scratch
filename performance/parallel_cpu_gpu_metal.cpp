/*
clang++ -std=c++17 parallel_cpu_gpu_metal.cpp \
  -Xpreprocessor -fopenmp \
  -I/usr/local/opt/libomp/include \
  -L/usr/local/opt/libomp/lib -lomp \
  -framework Foundation -framework Metal -o parallel_cpu_gpu_metal

./parallel_cpu_gpu_metal
*/

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "/Users/shivam/Open-Source/metal_deps/metal-cpp/SingleHeader/Metal.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

int main() {
    const int DATA_SIZE = 64 * 1024 * 1024; // 64M elements
    const int ITERATIONS = 20; 

    std::cout << "Allocating memory buffers (Total ~768MB Host RAM)..." << std::endl;
    std::vector<float> h_A(DATA_SIZE, 1.05f);
    std::vector<float> h_B(DATA_SIZE, 0.95f);
    std::vector<float> h_C_cpu(DATA_SIZE, 0.0f);

    // ==========================================
    // PHASE 1: MULTI-CORE CPU BENCHMARK
    // ==========================================
    std::cout << "\n[1/2] Starting MULTI-CORE CPU Benchmark (" << ITERATIONS << " passes)..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < ITERATIONS; step++) {
        // This pragma slices the 64M element iteration spaces across all available CPU threads automatically!
        #pragma omp parallel for
        for (int id = 0; id < DATA_SIZE; id++) {
            float a = h_A[id];
            float b = h_B[id];
            float res = 0.0f;
            for (int i = 0; i < 50; i++) {
                res += (a * b) + 0.01f;
            }
            h_C_cpu[id] = res;
        }
        std::cout << "  -> Multi-Core CPU Completed pass " << step + 1 << "/" << ITERATIONS << "..." << std::endl;
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Phase Complete. Sample element C_CPU[0]: " << h_C_cpu[0] << std::endl;
    std::cout << "Multi-Core CPU Time: " << cpu_duration.count() << " seconds." << std::endl;

    // ==========================================
    // PHASE 2: METAL GPU BENCHMARK
    // ==========================================
    std::cout << "\n[2/2] Initializing Native Metal runtime..." << std::endl;
    
    NS::Array* allDevices = MTL::CopyAllDevices();
    if (!allDevices || allDevices->count() == 0) {
        std::cerr << "CRITICAL ERROR: No Metal compatible graphics cards found at all!" << std::endl;
        return 1;
    }

    MTL::Device* device = nullptr;
    for (NS::UInteger i = 0; i < allDevices->count(); ++i) {
        MTL::Device* dev = static_cast<MTL::Device*>(allDevices->object(i));
        std::string devName(dev->name()->utf8String());
        if (devName.find("AMD") != std::string::npos || devName.find("Radeon") != std::string::npos) {
            device = dev;
        }
    }

    if (!device) {
        device = static_cast<MTL::Device*>(allDevices->object(0));
    }
    
    device->retain();
    allDevices->release();

    std::cout << "Targeting GPU for Benchmark: " << device->name()->utf8String() << std::endl;

    std::ifstream t("kernel.metal");
    if (!t.is_open()) {
        std::cerr << "ERROR: Could not open kernel.metal! Ensure it sits in the same directory.\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    NS::String* sourceCode = NS::String::string(buffer.str().c_str(), NS::UTF8StringEncoding);

    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(sourceCode, nullptr, &error);
    if (!library) {
        std::cerr << "Shader Compilation Error: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    NS::String* functionName = NS::String::string("vector_stress", NS::UTF8StringEncoding);
    MTL::Function* kernelFunc = library->newFunction(functionName);
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(kernelFunc, &error);
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    size_t bufferSize = sizeof(float) * DATA_SIZE;
    MTL::Buffer* d_A = device->newBuffer(h_A.data(), bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_B = device->newBuffer(h_B.data(), bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* d_C = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

    std::cout << "Executing Metal GPU loop (" << ITERATIONS << " passes)..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < ITERATIONS; step++) {
        MTL::CommandBuffer* cmdBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(d_A, 0, 0);
        encoder->setBuffer(d_B, 0, 1);
        encoder->setBuffer(d_C, 0, 2);

        NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > 256) threadGroupSize = 256;

        MTL::Size gridSize = MTL::Size(DATA_SIZE, 1, 1);
        MTL::Size groupSize = MTL::Size(threadGroupSize, 1, 1);

        encoder->dispatchThreads(gridSize, groupSize);
        encoder->endEncoding();
        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;

    float* gpu_res_ptr = (float*)d_C->contents();
    std::cout << "Metal Phase Complete. Sample element C_GPU[0]: " << gpu_res_ptr[0] << std::endl;
    std::cout << "Metal GPU Time: " << gpu_duration.count() << " seconds." << std::endl;

    // ==========================================
    // PHASE 3: COMPARISON SUMMARY
    // ==========================================
    std::cout << "\n============================================\n";
    std::cout << "     METAL VS MULTI-CORE CPU BENCHMARK       \n";
    std::cout << "============================================\n";
    std::cout << "Multi-Core CPU Runtime : " << cpu_duration.count() << " seconds\n";
    std::cout << "Metal GPU Runtime      : " << gpu_duration.count() << " seconds\n";
    
    double speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "--------------------------------------------\n";
    std::cout << "RESULT: Metal was " << speedup << "x FASTER than Multi-Core CPU!\n";
    std::cout << "============================================\n";

    d_A->release(); d_B->release(); d_C->release();
    commandQueue->release(); pipelineState->release(); kernelFunc->release(); library->release();
    device->release();
    return 0;
}
