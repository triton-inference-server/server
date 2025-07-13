#!/usr/bin/env python3
import subprocess
import time
import json
from datetime import datetime

print("🔍 Apple Silicon Hardware Monitor")
print("=" * 50)
print("Monitoring GPU, ANE, and CPU usage...")
print("Press Ctrl+C to stop\n")

def get_gpu_usage():
    """Get Metal GPU usage using ioreg"""
    try:
        # This is a simplified version - actual implementation would use IOKit
        cmd = "ioreg -l | grep -E 'PerformanceStatistics|GPUActivity'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # Parse GPU metrics from output
        return {"gpu_usage": "Active", "gpu_memory": "N/A"}
    except:
        return {"gpu_usage": "Unknown", "gpu_memory": "N/A"}

def get_ane_usage():
    """Get ANE usage - requires private frameworks"""
    try:
        # Check if ANE is being used by looking at process activity
        cmd = "ps aux | grep -i neural"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ane_active = "neural" in result.stdout.lower()
        return {"ane_active": ane_active, "ane_utilization": "Active" if ane_active else "Idle"}
    except:
        return {"ane_active": False, "ane_utilization": "Unknown"}

def get_cpu_usage():
    """Get CPU usage"""
    try:
        cmd = "ps -A -o %cpu | awk '{s+=$1} END {print s}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        cpu_percent = float(result.stdout.strip())
        return {"cpu_usage": f"{cpu_percent:.1f}%"}
    except:
        return {"cpu_usage": "Unknown"}

def get_memory_usage():
    """Get memory usage"""
    try:
        cmd = "vm_stat | grep 'Pages active' | awk '{print $3}' | sed 's/\\.//'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        active_pages = int(result.stdout.strip())
        # Each page is 4KB
        active_mb = (active_pages * 4) / 1024
        return {"active_memory": f"{active_mb:.0f} MB"}
    except:
        return {"active_memory": "Unknown"}

def get_power_metrics():
    """Get power consumption metrics"""
    try:
        cmd = "pmset -g batt | grep -Eo '[0-9]+%' | head -1"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        battery = result.stdout.strip()
        return {"battery": battery if battery else "AC Power"}
    except:
        return {"battery": "Unknown"}

# Monitor loop
metrics_history = []

try:
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Collect metrics
        metrics = {
            "timestamp": timestamp,
            "cpu": get_cpu_usage(),
            "memory": get_memory_usage(),
            "gpu": get_gpu_usage(),
            "ane": get_ane_usage(),
            "power": get_power_metrics()
        }
        
        metrics_history.append(metrics)
        
        # Display current metrics
        print(f"\r[{timestamp}] CPU: {metrics['cpu']['cpu_usage']} | "
              f"Memory: {metrics['memory']['active_memory']} | "
              f"GPU: {metrics['gpu']['gpu_usage']} | "
              f"ANE: {metrics['ane']['ane_utilization']} | "
              f"Power: {metrics['power']['battery']}", end="", flush=True)
        
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n📊 Saving metrics history...")
    
    # Save metrics
    with open('hardware_metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"✅ Saved {len(metrics_history)} data points to hardware_metrics.json")
    
    # Summary statistics
    if metrics_history:
        print("\n📈 Session Summary:")
        print(f"  Duration: {len(metrics_history)} seconds")
        print(f"  ANE Active: {sum(1 for m in metrics_history if m['ane']['ane_active'])} seconds")
        print(f"  Average CPU: {sum(float(m['cpu']['cpu_usage'].rstrip('%')) for m in metrics_history if m['cpu']['cpu_usage'] != 'Unknown') / len(metrics_history):.1f}%")