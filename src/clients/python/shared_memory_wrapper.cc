#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
// #include "src/clients/c++/shared_memory_wrapper.h"

#ifdef __cplusplus

extern "C" {
#endif

int
CreateSharedMemoryRegion(const char* shm_key, size_t batch_byte_size)
{
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return -1;
  }
  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(shm_fd, batch_byte_size);
  if (res == -1) {
    return -2;
  }
  return shm_fd;
}

int
OpenSharedMemoryRegion(const char* shm_key)
{
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key, O_RDWR, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return -1;
  }

  return shm_fd;
}

int
CloseSharedMemoryRegion(int shm_fd)
{
  int tmp = close(shm_fd);
  if (tmp == -1) {
    return -1;
  }
  return tmp;
}

void*
MapSharedMemory(int shm_fd, size_t offset, size_t batch_byte_size)
{
  // map shared memory to process address space
  void* shm_addr =
      mmap(NULL, batch_byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (shm_addr == MAP_FAILED) {
    return nullptr;
  }
  return shm_addr;
}

int
UnlinkSharedMemoryRegion(const char* shm_key)
{
  int shm_fd = shm_unlink(shm_key);
  if (shm_fd == -1) {
    return -1;
  }
  return shm_fd;
}

int
UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  int tmp_fd = munmap(shm_addr, byte_size);
  if (tmp_fd == -1) {
    return -1;
  }
  return tmp_fd;
}

#ifdef __cplusplus
}
#endif
