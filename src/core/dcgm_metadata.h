#pragma once
#include <dcgm_agent.h>
#include <iostream>

struct DcgmMetadata {
  // DCGM handles for initialization and destruction
  dcgmHandle_t dcgm_handle_ = 0;
  dcgmGpuGrp_t groupId_ = 0;
  // DCGM Flags
  bool standalone_ = false;
  // DCGM Fields
  size_t field_count_ = 0;
  std::vector<unsigned short> fields_;
  // GPU Device Mapping
  std::map<uint32_t, uint32_t> cuda_ids_to_dcgm_ids_;
  std::vector<uint32_t> available_cuda_gpu_ids_;
  // Stop attempting metrics if they fail multiple consecutive
  // times for a device.
  const int fail_threshold_ = 3;
  // DCGM Failure Tracking
  std::vector<int> power_limit_fail_cnt_;
  std::vector<int> power_usage_fail_cnt_;
  std::vector<int> energy_fail_cnt_;
  std::vector<int> util_fail_cnt_;
  std::vector<int> mem_fail_cnt_;
  // DCGM Energy Tracking
  std::vector<unsigned long long> last_energy_;
};
