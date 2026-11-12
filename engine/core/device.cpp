#include "kvbound/cuda/device.h"

#include <string>

namespace kvb::cuda {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw Error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes > 0) KVB_CUDA(cudaMalloc(&data_, bytes));
}

DeviceBuffer::~DeviceBuffer() {
  if (data_ != nullptr) cudaFree(data_);
}

PinnedBuffer::PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes > 0) KVB_CUDA(cudaMallocHost(&data_, bytes));
}

PinnedBuffer::~PinnedBuffer() {
  if (data_ != nullptr) cudaFreeHost(data_);
}

}  // namespace kvb::cuda
