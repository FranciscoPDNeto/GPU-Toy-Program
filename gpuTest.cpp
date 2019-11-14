#include <CL/cl.hpp>

#include <cassert>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    return 1;
  }

  cl::Platform defaultPlatform = platforms[1];
  std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>()
            << "\n";

  std::cout << "Getting devices...";

  std::vector<cl::Device> devices;
  defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    return 1;
  }

  std::cout << "Done!" << std::endl;

  for (std::size_t i = 0; i < devices.size(); ++i) {
    std::cout << "Device " << i << " - " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
  }

  std::cout << "Select your device id: ";
  int deviceId;
  std::cin >> deviceId;
  cl::Device defaultDevice = devices[deviceId];
  std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << "\n";

  std::cout << "Creating context...";

  cl::Context context({ defaultDevice });

  std::cout << "Done!" << std::endl;

  cl::Program::Sources sources;
  const std::string kernelSource = 
    "void kernel sumBuffers(global const double* A, "
    "  global const double* B, global double* C) {\n"
    "\n"
    "  unsigned long taskIndex = get_global_id(0);\n"
    "  C[taskIndex] = A[taskIndex] + B[taskIndex];\n"
    "}";

  std::cout << "Building kernel...";

  sources.push_back({ kernelSource.c_str(), kernelSource.length() });
  cl::Program program(context, sources);
  if (program.build({ defaultDevice }) != CL_SUCCESS) {
    std::cout << "Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice)
              << std::endl;
    exit(1);
  }

  std::cout << "Done!" << std::endl;

  const std::size_t totalSize = 10;
  std::vector<double> numbers(totalSize, 0);
  for (double i = 0; i < totalSize; ++i)
    numbers[static_cast<int>(i)] = i * 0.5;
  
  std::cout << "Allocating buffers...";

  cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(double) * totalSize, nullptr);
  std::cout << "A done! ";
  cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(double) * totalSize, nullptr);
  std::cout << "B done! ";
  cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(double) * totalSize, nullptr);

  std::cout << "Done!" << std::endl;
  std::cout << "Running...";

  cl::CommandQueue queue(context, defaultDevice);
  queue.enqueueWriteBuffer(
    bufferA, CL_TRUE, 0, sizeof(double) * totalSize, numbers.data());
  queue.enqueueWriteBuffer(
    bufferB, CL_TRUE, 0, sizeof(double) * totalSize, numbers.data());

  cl::Kernel sumBuffers(program, "sumBuffers");
  sumBuffers.setArg(0, bufferA);
  sumBuffers.setArg(1, bufferB);
  sumBuffers.setArg(2, bufferC);

  queue.enqueueNDRangeKernel(sumBuffers, 0, totalSize, 32);

  std::cout << "Done!" << std::endl;
  std::cout << "Verifying results...";

  queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(double) * totalSize, numbers.data());
  for (std::size_t i = 0; i < totalSize; ++i) {
    if (numbers[i] != i) {
      std::cout << "Verification failed! result #" << i << ", " << 
        numbers[i] << " != " << i << " (expected)." << std::endl;

      return 0;
    }
  }

  std::cout << "Good! First 10 numbers: " << std::endl;
  for (std::size_t i = 0; i < 10; ++i) {
    std::cout << numbers[i] << std::endl;
  }

  return 0;
}
