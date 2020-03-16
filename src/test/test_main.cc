#include "gtest/gtest.h"

#include "src/core/cuda_memory_manager.h"

namespace {

TEST(CudaMemoryManager, Init) {
  double cc = 6.0;
  std::map<int, uint64_t> s{{0, 1024}};
  const nvidia::inferenceserver::CudaMemoryManager::Options options{cc, s};
  auto status = nvidia::inferenceserver::CudaMemoryManager::Create(options);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  void* ptr;
  status = nvidia::inferenceserver::CudaMemoryManager::Alloc(&ptr, 256, 0);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

}  // namespace