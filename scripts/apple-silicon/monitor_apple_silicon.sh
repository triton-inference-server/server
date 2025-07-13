#!/bin/bash
# Monitor Apple Silicon performance during inference

echo "🍎 Apple Silicon Performance Monitor"
echo "==================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to monitor ANE
monitor_ane() {
    echo "📊 Monitoring Neural Engine (ANE)..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    if command_exists powermetrics; then
        # Need sudo for powermetrics
        echo "⚠️  Note: powermetrics requires sudo access"
        sudo powermetrics --samplers tasks,gpu_power -i 1000 -n 10 | grep -E "(Neural|tasks|Package Power)"
    else
        echo "❌ powermetrics not found. Using alternative method..."
        # Alternative: Use ioreg to check ANE
        while true; do
            echo -n "$(date '+%H:%M:%S') - "
            ioreg -l | grep -i "neural" | head -5
            sleep 1
        done
    fi
}

# Function to monitor Metal GPU
monitor_metal() {
    echo "🎮 Monitoring Metal GPU..."
    echo ""
    
    if command_exists asitop; then
        echo "Using asitop (if installed via pip install asitop)..."
        asitop
    else
        echo "💡 Tip: Install asitop for better monitoring: pip install asitop"
        echo ""
        # Use Activity Monitor data
        while true; do
            echo "$(date '+%H:%M:%S') GPU Usage:"
            top -l 1 -s 0 | grep -E "(tritonserver|Metal|GPU)" | head -10
            echo "---"
            sleep 1
        done
    fi
}

# Function to monitor memory
monitor_memory() {
    echo "🧠 Monitoring Memory Usage..."
    echo ""
    
    while true; do
        clear
        echo "🍎 Apple Silicon Memory Monitor - $(date '+%H:%M:%S')"
        echo "============================================"
        
        # Overall memory
        vm_stat | grep -E "(free|active|inactive|wired|compressed)"
        
        echo ""
        echo "Triton Server Memory:"
        ps aux | grep -E "(USER|tritonserver)" | grep -v grep
        
        echo ""
        echo "Top Processes by Memory:"
        top -l 1 -o mem -n 5 | tail -6
        
        sleep 2
    done
}

# Function to monitor all metrics
monitor_all() {
    echo "📈 Comprehensive Apple Silicon Monitoring"
    echo ""
    
    # Create a simple dashboard
    while true; do
        clear
        echo "🍎 Apple Silicon Performance Dashboard - $(date '+%H:%M:%S')"
        echo "=================================================="
        
        # CPU Usage
        echo "🖥️  CPU Usage:"
        top -l 1 | grep "CPU usage" | cut -d' ' -f3-
        
        # Memory
        echo ""
        echo "🧠 Memory:"
        memory_pressure | grep -E "(Free|Pressure)" | head -2
        
        # GPU (if available)
        echo ""
        echo "🎮 GPU:"
        ioreg -r -d 1 -w 0 -c "IOAccelerator" | grep -E "(PerformanceStatistics|DeviceUsage)" | head -5
        
        # Thermal
        echo ""
        echo "🌡️  Thermal State:"
        pmset -g therm | tail -1
        
        # Power
        echo ""
        echo "⚡ Power:"
        pmset -g batt | grep -E "(AC|Battery)" | head -1
        
        # Triton Process
        echo ""
        echo "🚀 Triton Server:"
        ps aux | grep tritonserver | grep -v grep | awk '{print "CPU: "$3"% MEM: "$4"% TIME: "$10}'
        
        sleep 2
    done
}

# Function to show Triton metrics
show_triton_metrics() {
    echo "📊 Triton Server Metrics"
    echo ""
    
    if curl -s http://localhost:8002/metrics > /dev/null 2>&1; then
        echo "Apple Silicon Specific Metrics:"
        curl -s http://localhost:8002/metrics | grep -E "(apple_silicon|ane_|metal_|amx_)" | grep -v "^#"
        
        echo ""
        echo "Inference Metrics:"
        curl -s http://localhost:8002/metrics | grep -E "(nv_inference_request_success|nv_inference_request_duration|nv_gpu_utilization)" | grep -v "^#" | head -10
    else
        echo "❌ Metrics server not available. Make sure Triton is running with --allow-metrics=true"
    fi
}

# Main menu
PS3='Select monitoring option: '
options=(
    "Monitor Neural Engine (ANE)"
    "Monitor Metal GPU"
    "Monitor Memory Usage"
    "Monitor All Metrics"
    "Show Triton Metrics"
    "Launch Activity Monitor"
    "Quit"
)

select opt in "${options[@]}"
do
    case $opt in
        "Monitor Neural Engine (ANE)")
            monitor_ane
            ;;
        "Monitor Metal GPU")
            monitor_metal
            ;;
        "Monitor Memory Usage")
            monitor_memory
            ;;
        "Monitor All Metrics")
            monitor_all
            ;;
        "Show Triton Metrics")
            show_triton_metrics
            ;;
        "Launch Activity Monitor")
            open -a "Activity Monitor"
            echo "✅ Activity Monitor launched. Look for tritonserver process."
            ;;
        "Quit")
            break
            ;;
        *) echo "Invalid option $REPLY";;
    esac
done