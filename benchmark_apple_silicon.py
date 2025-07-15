#!/usr/bin/env python3
"""
Benchmark script for Apple Silicon optimizations in Triton
Tests matrix multiplication performance with AMX acceleration
"""

import numpy as np
import time
import os
import sys
import subprocess
from typing import List, Tuple, Dict

# Try to import Accelerate framework bindings if available
try:
    from accelerate import Framework
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

class AppleSiliconBenchmark:
    def __init__(self):
        self.sizes = [256, 512, 1024, 2048, 4096]
        self.iterations = 10
        self.warmup_iterations = 3
        
    def benchmark_numpy_matmul(self, size: int) -> Tuple[float, float]:
        """Benchmark NumPy matrix multiplication (uses Accelerate on macOS)"""
        # Create random matrices
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = np.matmul(a, b)
        
        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            c = np.matmul(a, b)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate GFLOPS
        flops = 2 * size * size * size  # 2 * M * N * K
        gflops = flops / (avg_time * 1e9)
        
        return avg_time, gflops
    
    def benchmark_direct_amx(self, size: int) -> Tuple[float, float]:
        """Benchmark direct AMX calls through our provider"""
        # This would use our AMX provider directly
        # For now, we'll use NumPy as a proxy since it uses Accelerate
        return self.benchmark_numpy_matmul(size)
    
    def check_amx_availability(self) -> bool:
        """Check if AMX is available on this system"""
        try:
            # Check CPU features
            result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                                  capture_output=True, text=True)
            has_arm64 = result.returncode == 0 and result.stdout.strip() == '1'
            
            if has_arm64:
                # Check for AMX feature
                result = subprocess.run(['sysctl', 'hw.optional.amx'], 
                                      capture_output=True, text=True)
                return 'amx_version' in result.stdout or result.returncode == 0
        except:
            pass
        return False
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("Apple Silicon Optimization Benchmark")
        print("=" * 60)
        
        # Check system
        has_amx = self.check_amx_availability()
        print(f"AMX Available: {has_amx}")
        print(f"NumPy version: {np.__version__}")
        print(f"NumPy BLAS: {np.show_config()}")
        print()
        
        # Run benchmarks for different sizes
        results = []
        for size in self.sizes:
            print(f"\nMatrix Size: {size}x{size}")
            print("-" * 40)
            
            # NumPy benchmark (uses Accelerate/AMX on macOS)
            np_time, np_gflops = self.benchmark_numpy_matmul(size)
            print(f"NumPy (Accelerate): {np_gflops:.2f} GFLOPS ({np_time*1000:.2f} ms)")
            
            results.append({
                'size': size,
                'numpy_time': np_time,
                'numpy_gflops': np_gflops
            })
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"{'Size':>6} | {'NumPy (GFLOPS)':>15} | {'Time (ms)':>10}")
        print("-" * 40)
        for r in results:
            print(f"{r['size']:>6} | {r['numpy_gflops']:>15.2f} | {r['numpy_time']*1000:>10.2f}")
        
        # Performance characteristics
        print("\nPerformance Analysis:")
        print("-" * 40)
        
        # Check if performance scales with AMX tile sizes (32x32)
        if len(results) >= 3:
            # Compare small vs large matrix efficiency
            small_eff = results[0]['numpy_gflops']
            large_eff = results[-1]['numpy_gflops']
            
            print(f"Small matrix (256x256) efficiency: {small_eff:.2f} GFLOPS")
            print(f"Large matrix ({self.sizes[-1]}x{self.sizes[-1]}) efficiency: {large_eff:.2f} GFLOPS")
            print(f"Scaling factor: {large_eff/small_eff:.2f}x")
            
            # Theoretical peak for M1/M2 is ~2000 GFLOPS for FP32
            theoretical_peak = 2000  # Approximate for Apple Silicon
            peak_efficiency = (large_eff / theoretical_peak) * 100
            print(f"Peak efficiency: {peak_efficiency:.1f}% of theoretical")

def test_amx_kernel():
    """Test our AMX kernel implementation directly"""
    print("\nTesting AMX Kernel Implementation")
    print("=" * 60)
    
    # This would test our actual AMX kernels
    # For now, we'll create a simple test
    size = 1024
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Test different data types
    print("\nData Type Performance:")
    print("-" * 40)
    
    # FP32
    start = time.perf_counter()
    c_fp32 = np.matmul(a, b)
    fp32_time = time.perf_counter() - start
    fp32_gflops = (2 * size * size * size) / (fp32_time * 1e9)
    print(f"FP32: {fp32_gflops:.2f} GFLOPS")
    
    # FP16 (if supported)
    try:
        a_fp16 = a.astype(np.float16)
        b_fp16 = b.astype(np.float16)
        start = time.perf_counter()
        c_fp16 = np.matmul(a_fp16, b_fp16)
        fp16_time = time.perf_counter() - start
        fp16_gflops = (2 * size * size * size) / (fp16_time * 1e9)
        print(f"FP16: {fp16_gflops:.2f} GFLOPS ({fp16_gflops/fp32_gflops:.1f}x speedup)")
    except:
        print("FP16: Not supported")

if __name__ == "__main__":
    benchmark = AppleSiliconBenchmark()
    benchmark.run_benchmarks()
    test_amx_kernel()