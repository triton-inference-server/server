#include "gtest/gtest.h"

#include <cuda_runtime_api.h>
#include "src/core/cuda_memory_manager.h"
#include "src/core/cuda_utils.h"

namespace ni = nvidia::inferenceserver;

namespace {

#define CHECK_POINTER_ATTRIBUTES(ptr__, type__, device__)                   \
  do {                                                                      \
    cudaPointerAttributes attr;                                             \
    auto cuerr = cudaPointerGetAttributes(&attr, ptr__);                    \
    ASSERT_TRUE(cuerr == cudaSuccess)                                       \
        << "Failed to get CUDA pointer attributes: "                        \
        << cudaGetErrorString(cuerr);                                       \
    EXPECT_TRUE(attr.type == type__)                                        \
        << "Expect pointer with type " << type__ << ", got: " << attr.type; \
    if (attr.type == cudaMemoryTypeDevice) {                                \
      EXPECT_TRUE(attr.device == device__)                                  \
          << "Expect allocation on CUDA device " << device__                \
          << ", got: " << attr.device;                                      \
    }                                                                       \
  } while (false)

// Wrapper of CudaMemoryManager class to expose Reset() for unit testing
class TestingCudaMemoryManager : public ni::CudaMemoryManager {
 public:
  static void Reset() { CudaMemoryManager::Reset(); }
};

class CudaMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Default memory manager options
    options_.min_supported_compute_capability_ = 6.0;
    options_.memory_pool_byte_size_ = {{0, 1 << 10}};
  }

  void TearDown() override { TestingCudaMemoryManager::Reset(); }

  ni::CudaMemoryManager::Options options_;
};

TEST_F(CudaMemoryManagerTest, Init)
{
  // Set to reserve too much memory
  {
    double cc = 6.0;
    std::map<int, uint64_t> s{{0, uint64_t(1) << 40 /* 1024 GB */}};
    const ni::CudaMemoryManager::Options options{cc, s};
    auto status = ni::CudaMemoryManager::Create(options);
    EXPECT_FALSE(status.IsOk()) << "Expect creation error";
  }

  {
    double cc = 6.0;
    std::map<int, uint64_t> s{{0, 1 << 10 /* 1024 bytes */}};
    const ni::CudaMemoryManager::Options options{cc, s};
    auto status = ni::CudaMemoryManager::Create(options);
    EXPECT_TRUE(status.IsOk()) << status.Message();
  }
}

TEST_F(CudaMemoryManagerTest, AllocSuccess)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, 0);
}

TEST_F(CudaMemoryManagerTest, AllocFail)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&ptr, 2048, 0);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";
}

TEST_F(CudaMemoryManagerTest, MultipleAlloc)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* first_ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&first_ptr, 600, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(first_ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(first_ptr, cudaMemoryTypeDevice, 0);

  // 512 + 600 > 1024
  void* second_ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&second_ptr, 512, 0);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";

  // Free the first pointer and retry the second one
  status = ni::CudaMemoryManager::Free(first_ptr, 0);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  status = ni::CudaMemoryManager::Alloc(&second_ptr, 512, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(second_ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(second_ptr, cudaMemoryTypeDevice, 0);
}

TEST_F(CudaMemoryManagerTest, MultipleDevice)
{
  std::set<int> supported_gpus;
  auto status = ni::GetSupportedGPUs(
      &supported_gpus, options_.min_supported_compute_capability_);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_GE(supported_gpus.size(), size_t(2))
      << "Test requires at least two supported CUDA devices";

  {
    double cc = 6.0;
    std::map<int, uint64_t> s;
    // Only enough memory is only reserved in one of the devices
    s[*supported_gpus.begin()] = 32;
    s[*(++supported_gpus.begin())] = 1024;
    const ni::CudaMemoryManager::Options options{cc, s};
    status = ni::CudaMemoryManager::Create(options);
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  void* ptr = nullptr;
  // Allocation on small device
  int small_device = *supported_gpus.begin();
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, small_device);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";

  // Allocation on large device
  int large_device = *(++supported_gpus.begin());
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, large_device);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, large_device);

  // Free allocation ...
  status = ni::CudaMemoryManager::Free(ptr, small_device);
  EXPECT_FALSE(status.IsOk()) << "Unexpected deallocation on wrong device";
  status = ni::CudaMemoryManager::Free(ptr, large_device);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

}  // namespace