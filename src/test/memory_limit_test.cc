// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gtest/gtest.h"

// Undefine the FAIL() macro inside Triton code to avoid redefine error
// from gtest. Okay as FAIL() is not used in these tests.
#ifdef FAIL
#undef FAIL
#endif

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "memory_limit.h"

namespace tsrv = triton::server;

namespace {

constexpr uint64_t kMiB = 1ULL << 20;
constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kRam = 256 * kGiB;
const std::string kV1NoLimit = "9223372036854771712";
const std::string kV2Mount =
    "30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - "
    "cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n";
const std::string kV1DockerMount =
    "40 30 0:35 /docker/abc /sys/fs/cgroup/memory ro,nosuid master:16 - "
    "cgroup cgroup rw,memory\n";

// Builds a fake /proc and /sys tree and reads the memory limit from it.
class MemoryLimitTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    root_ = std::filesystem::temp_directory_path().string() +
            "/memory_limit_test_" + info->name();
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  // Writes 'contents' to 'path' (absolute inside the fake tree).
  void Write(const std::string& path, const std::string& contents)
  {
    const std::filesystem::path full =
        std::filesystem::path(root_) / path.substr(1);
    std::filesystem::create_directories(full.parent_path());
    std::ofstream(full) << contents;
  }

  // Marks 'dir' as an existing cgroup directory.
  void Cgroup(const std::string& dir) { Write(dir + "/cgroup.procs", "1\n"); }

  tsrv::MemoryLimit Detect(uint64_t physical_ram = kRam)
  {
    return tsrv::DetectMemoryLimit(root_, physical_ram);
  }

  std::string root_;
};

TEST_F(MemoryLimitTest, V2ContainerWithLimit)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "1073741824\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 1 * kGiB);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::CGROUP_V2);
  EXPECT_EQ(limit.detail, "/sys/fs/cgroup/memory.max");
  EXPECT_FALSE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, V2ContainerWithoutLimit)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "max\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, kRam);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
  EXPECT_EQ(limit.detail, "no cgroup memory limit set");
  EXPECT_FALSE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, NestedCgroupsWithoutLimit)
{
  // Outside a container: every level is "max" and the root has no file.
  const std::string scope = "/sys/fs/cgroup/user.slice/user-1.slice/s.scope";
  Write("/proc/self/cgroup", "0::/user.slice/user-1.slice/s.scope\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup(scope);
  Write(scope + "/memory.max", "max\n");
  Write("/sys/fs/cgroup/user.slice/user-1.slice/memory.max", "max\n");
  Write("/sys/fs/cgroup/user.slice/memory.max", "max\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, kRam);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
  EXPECT_FALSE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, HostCgroupNamespaceUsesOwnCgroup)
{
  const std::string scope = "/sys/fs/cgroup/system.slice/docker-abc.scope";
  Write("/proc/self/cgroup", "0::/system.slice/docker-abc.scope\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup(scope);
  Write(scope + "/memory.max", "1073741824\n");
  Write("/sys/fs/cgroup/system.slice/memory.max", "max\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 1 * kGiB);
  EXPECT_EQ(limit.detail, scope + "/memory.max");
}

TEST_F(MemoryLimitTest, SmallestLimitUpTheTreeWins)
{
  Write("/proc/self/cgroup", "0::/kubepods/pod1/ctr1\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup/kubepods/pod1/ctr1");
  Write("/sys/fs/cgroup/kubepods/pod1/ctr1/memory.max", "4294967296\n");
  Write("/sys/fs/cgroup/kubepods/pod1/memory.max", "2147483648\n");
  Write("/sys/fs/cgroup/kubepods/memory.max", "max\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 2 * kGiB);
  EXPECT_EQ(limit.detail, "/sys/fs/cgroup/kubepods/pod1/memory.max");
}

TEST_F(MemoryLimitTest, V1MountRootIsOwnCgroup)
{
  Write(
      "/proc/self/cgroup",
      "12:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n"
      "1:name=systemd:/docker/abc\n");
  Write(
      "/proc/self/mountinfo",
      kV1DockerMount +
          "41 30 0:36 /docker/abc /sys/fs/cgroup/cpu,cpuacct ro master:17 - "
          "cgroup cgroup rw,cpu,cpuacct\n");
  Cgroup("/sys/fs/cgroup/memory");
  Write("/sys/fs/cgroup/memory/memory.limit_in_bytes", "536870912\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 512 * kMiB);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::CGROUP_V1);
  EXPECT_EQ(limit.detail, "/sys/fs/cgroup/memory/memory.limit_in_bytes");
}

TEST_F(MemoryLimitTest, V1UnlimitedValueIsNoLimit)
{
  Write("/proc/self/cgroup", "12:memory:/docker/abc\n");
  Write("/proc/self/mountinfo", kV1DockerMount);
  Cgroup("/sys/fs/cgroup/memory");
  Write("/sys/fs/cgroup/memory/memory.limit_in_bytes", kV1NoLimit + "\n");

  EXPECT_EQ(Detect().source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
  // Also without knowing physical RAM.
  const auto limit = Detect(0 /* physical_ram */);
  EXPECT_EQ(limit.bytes, 0u);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
}

TEST_F(MemoryLimitTest, HybridHostUsesV1Memory)
{
  const std::string v1 = "/sys/fs/cgroup/memory";
  Write(
      "/proc/self/cgroup",
      "10:memory:/user.slice/x.scope\n0::/user.slice/x.scope\n");
  Write(
      "/proc/self/mountinfo",
      "30 25 0:26 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n"
      "40 25 0:35 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n");
  Cgroup(v1 + "/user.slice/x.scope");
  Write(v1 + "/user.slice/x.scope/memory.limit_in_bytes", "2147483648\n");
  Write(v1 + "/user.slice/memory.limit_in_bytes", kV1NoLimit + "\n");
  Write(v1 + "/memory.limit_in_bytes", kV1NoLimit + "\n");
  Cgroup("/sys/fs/cgroup/unified/user.slice/x.scope");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 2 * kGiB);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::CGROUP_V1);
}

TEST_F(MemoryLimitTest, EscapedSpaceInMountRoot)
{
  // mountinfo escapes the space; /proc/self/cgroup does not.
  Write("/proc/self/cgroup", "12:memory:/docker/my ctr\n");
  Write(
      "/proc/self/mountinfo",
      "40 30 0:35 /docker/my\\040ctr /sys/fs/cgroup/memory ro - cgroup "
      "cgroup rw,memory\n");
  Cgroup("/sys/fs/cgroup/memory");
  Write("/sys/fs/cgroup/memory/memory.limit_in_bytes", "536870912\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 512 * kMiB);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::CGROUP_V1);
}

TEST_F(MemoryLimitTest, EscapedSpaceInMountPoint)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write(
      "/proc/self/mountinfo",
      "30 25 0:26 / /sys/fs/cg\\040root rw - cgroup2 cgroup2 rw\n");
  Cgroup("/sys/fs/cg root");
  Write("/sys/fs/cg root/memory.max", "1073741824\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, 1 * kGiB);
  EXPECT_EQ(limit.detail, "/sys/fs/cg root/memory.max");
}

TEST_F(MemoryLimitTest, MissingOwnCgroupFallsBack)
{
  // A limit exists elsewhere but our own cgroup directory does not; never
  // report a limit that may belong to another cgroup.
  Write("/proc/self/cgroup", "0::/system.slice/docker-zzz.scope\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Write("/sys/fs/cgroup/memory.max", "1073741824\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, kRam);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
  EXPECT_TRUE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, NoCgroupInformationFallsBack)
{
  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, kRam);
  EXPECT_EQ(limit.detail, "no cgroup information");
  EXPECT_FALSE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, PathOutsideNamespaceFallsBack)
{
  Write("/proc/self/cgroup", "0::/../../system.slice/x.scope\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "1073741824\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
  EXPECT_TRUE(limit.detection_failed);
}

TEST_F(MemoryLimitTest, SimilarPrefixDoesNotMatchMountRoot)
{
  Write("/proc/self/cgroup", "12:memory:/docker/abcdef\n");
  Write("/proc/self/mountinfo", kV1DockerMount);
  Cgroup("/sys/fs/cgroup/memory");
  Write("/sys/fs/cgroup/memory/memory.limit_in_bytes", "536870912\n");

  EXPECT_EQ(Detect().source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
}

TEST_F(MemoryLimitTest, LimitAtOrAboveRamIsNoLimit)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", std::to_string(2 * kRam) + "\n");

  const auto limit = Detect();
  EXPECT_EQ(limit.bytes, kRam);
  EXPECT_EQ(limit.source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
}

TEST_F(MemoryLimitTest, UnparsableLimitIsIgnored)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "abc123\n");

  EXPECT_EQ(Detect().source, tsrv::MemoryLimitSource::PHYSICAL_RAM);
}

// The automatic budget, worked out from a detected limit.
TEST_F(MemoryLimitTest, AutomaticBudgetIsHalfOfCgroupLimit)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "1073741824\n");

  const auto budget = tsrv::ResolveParseMemoryBudget(
      tsrv::HTTP_PARSE_MEMORY_BUDGET_AUTO, Detect());
  EXPECT_TRUE(budget.enabled);
  EXPECT_EQ(budget.bytes, 512 * kMiB);
  EXPECT_FALSE(budget.warn);
  EXPECT_NE(
      budget.description.find("512 MiB (50% of the memory limit 1 GiB from "
                              "/sys/fs/cgroup/memory.max)"),
      std::string::npos)
      << budget.description;
}

TEST_F(MemoryLimitTest, AutomaticBudgetIsHalfOfRamWithoutLimit)
{
  Write("/proc/self/cgroup", "0::/\n");
  Write("/proc/self/mountinfo", kV2Mount);
  Cgroup("/sys/fs/cgroup");
  Write("/sys/fs/cgroup/memory.max", "max\n");

  const auto budget = tsrv::ResolveParseMemoryBudget(
      tsrv::HTTP_PARSE_MEMORY_BUDGET_AUTO, Detect());
  EXPECT_TRUE(budget.enabled);
  EXPECT_EQ(budget.bytes, 128 * kGiB);
  EXPECT_FALSE(budget.warn);
  EXPECT_NE(
      budget.description.find("128 GiB (50% of physical RAM 256 GiB"),
      std::string::npos)
      << budget.description;
}

TEST_F(MemoryLimitTest, AutomaticBudgetWarnsWhenCgroupNotResolved)
{
  Write("/proc/self/cgroup", "0::/system.slice/docker-zzz.scope\n");
  Write("/proc/self/mountinfo", kV2Mount);

  const auto budget = tsrv::ResolveParseMemoryBudget(
      tsrv::HTTP_PARSE_MEMORY_BUDGET_AUTO, Detect());
  EXPECT_TRUE(budget.enabled);
  EXPECT_EQ(budget.bytes, 128 * kGiB);
  EXPECT_TRUE(budget.warn);
}

}  // namespace

// Budget resolution from a given limit.
TEST(ResolveParseMemoryBudgetTest, AutomaticBudgetOffWhenSizeUnknown)
{
  tsrv::MemoryLimit unknown;
  unknown.bytes = 0;
  const auto budget = tsrv::ResolveParseMemoryBudget(
      tsrv::HTTP_PARSE_MEMORY_BUDGET_AUTO, unknown);
  EXPECT_FALSE(budget.enabled);
  EXPECT_TRUE(budget.warn);
}

TEST(ResolveParseMemoryBudgetTest, AutomaticBudgetForVeryLargeLimit)
{
  tsrv::MemoryLimit huge;
  huge.bytes = (1ULL << 60) - 1;
  huge.source = tsrv::MemoryLimitSource::CGROUP_V2;
  const auto budget =
      tsrv::ResolveParseMemoryBudget(tsrv::HTTP_PARSE_MEMORY_BUDGET_AUTO, huge);
  EXPECT_EQ(budget.bytes, ((1ULL << 60) - 1) / 2);
}

TEST(ResolveParseMemoryBudgetTest, ZeroTurnsBudgetOff)
{
  const auto budget = tsrv::ResolveParseMemoryBudget(0, tsrv::MemoryLimit{});
  EXPECT_FALSE(budget.enabled);
  EXPECT_TRUE(budget.warn);
}

TEST(ResolveParseMemoryBudgetTest, FlagValueIsUsedAsIs)
{
  tsrv::MemoryLimit limit;
  limit.bytes = 1 * kGiB;
  limit.source = tsrv::MemoryLimitSource::CGROUP_V2;
  const auto budget = tsrv::ResolveParseMemoryBudget(256 * kMiB, limit);
  EXPECT_TRUE(budget.enabled);
  EXPECT_EQ(budget.bytes, 256 * kMiB);
  EXPECT_FALSE(budget.warn);
}

TEST(ResolveParseMemoryBudgetTest, FlagAboveCgroupLimitWarns)
{
  tsrv::MemoryLimit limit;
  limit.bytes = 1 * kGiB;
  limit.source = tsrv::MemoryLimitSource::CGROUP_V2;
  const auto budget = tsrv::ResolveParseMemoryBudget(2 * kGiB, limit);
  EXPECT_EQ(budget.bytes, 2 * kGiB);
  EXPECT_TRUE(budget.warn);
}

TEST(ResolveParseMemoryBudgetTest, FlagWithoutCgroupLimitDoesNotWarn)
{
  tsrv::MemoryLimit ram;
  ram.bytes = kRam;
  ram.source = tsrv::MemoryLimitSource::PHYSICAL_RAM;
  const auto budget = tsrv::ResolveParseMemoryBudget(4 * kGiB, ram);
  EXPECT_FALSE(budget.warn);
}
