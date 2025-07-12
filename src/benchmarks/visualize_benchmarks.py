#!/usr/bin/env python3
"""
Visualize Apple Silicon benchmark results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkVisualizer:
    def __init__(self, results_path):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path.parent / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        if results_path.endswith('.json'):
            self.load_json()
        else:
            self.load_csv()
    
    def load_json(self):
        with open(self.results_path) as f:
            data = json.load(f)
        self.df = pd.DataFrame(data['results'])
    
    def load_csv(self):
        self.df = pd.read_csv(self.results_path)
    
    def plot_gemm_performance(self):
        """Plot GEMM performance comparison"""
        gemm_df = self.df[self.df['operation'] == 'GEMM'].copy()
        
        # Extract matrix size for sorting
        gemm_df['matrix_size'] = gemm_df['dimensions'].apply(
            lambda x: eval(x)[0] if isinstance(x, str) else x[0]
        )
        gemm_df = gemm_df.sort_values('matrix_size')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance (GFLOPS) comparison
        ax = axes[0, 0]
        processors = gemm_df['processor'].unique()
        x = np.arange(len(gemm_df['dimensions'].unique()))
        width = 0.8 / len(processors)
        
        for i, proc in enumerate(processors):
            proc_data = gemm_df[gemm_df['processor'] == proc]
            ax.bar(x + i * width, proc_data['gflops'], width, label=proc)
        
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('GFLOPS')
        ax.set_title('GEMM Performance Comparison')
        ax.set_xticks(x + width * (len(processors) - 1) / 2)
        ax.set_xticklabels([str(d) for d in gemm_df['dimensions'].unique()], rotation=45)
        ax.legend()
        
        # 2. Speedup relative to CPU
        ax = axes[0, 1]
        cpu_perf = gemm_df[gemm_df['processor'] == 'CPU'].set_index('dimensions')['gflops']
        
        for proc in processors:
            if proc != 'CPU':
                proc_data = gemm_df[gemm_df['processor'] == proc].set_index('dimensions')
                speedup = proc_data['gflops'] / cpu_perf
                ax.plot(range(len(speedup)), speedup.values, marker='o', label=f'{proc} vs CPU')
        
        ax.set_xlabel('Matrix Size Index')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup Relative to CPU')
        ax.legend()
        ax.grid(True)
        
        # 3. Memory bandwidth utilization
        ax = axes[1, 0]
        for proc in processors:
            proc_data = gemm_df[gemm_df['processor'] == proc]
            ax.plot(proc_data['matrix_size'], proc_data['gb_per_sec'], 
                   marker='o', label=proc)
        
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('GB/s')
        ax.set_title('Memory Bandwidth Utilization')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True)
        
        # 4. Efficiency (GFLOPS/W)
        ax = axes[1, 1]
        efficiency_data = gemm_df[gemm_df['power_watts'] > 0]
        
        for proc in processors:
            proc_data = efficiency_data[efficiency_data['processor'] == proc]
            if not proc_data.empty:
                ax.bar(proc, proc_data['efficiency_gflops_per_watt'].mean())
        
        ax.set_ylabel('GFLOPS/Watt')
        ax.set_title('Power Efficiency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gemm_performance.png', dpi=300)
        plt.close()
    
    def plot_operation_comparison(self):
        """Compare different operations across processors"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by operation and processor
        grouped = self.df.groupby(['operation', 'processor'])['gflops'].mean().reset_index()
        pivot = grouped.pivot(index='operation', columns='processor', values='gflops')
        
        # Create grouped bar chart
        pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel('GFLOPS')
        ax.set_title('Performance by Operation Type')
        ax.legend(title='Processor')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'operation_comparison.png', dpi=300)
        plt.close()
    
    def plot_latency_distribution(self):
        """Plot latency distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        processors = self.df['processor'].unique()
        
        for i, proc in enumerate(processors):
            ax = axes[i // 2, i % 2]
            proc_data = self.df[self.df['processor'] == proc]
            
            # Violin plot for latency distribution
            operations = proc_data['operation'].unique()
            data_to_plot = [proc_data[proc_data['operation'] == op]['avg_time_ms'].values 
                           for op in operations]
            
            ax.violinplot(data_to_plot, showmeans=True)
            ax.set_xticks(range(1, len(operations) + 1))
            ax.set_xticklabels(operations, rotation=45)
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'{proc} Latency Distribution')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png', dpi=300)
        plt.close()
    
    def plot_power_efficiency_heatmap(self):
        """Create heatmap of power efficiency"""
        # Filter data with power measurements
        power_data = self.df[self.df['power_watts'] > 0].copy()
        
        if power_data.empty:
            print("No power data available for heatmap")
            return
        
        # Create pivot table
        pivot = power_data.pivot_table(
            values='efficiency_gflops_per_watt',
            index='operation',
            columns='processor',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'GFLOPS/Watt'})
        plt.title('Power Efficiency Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'power_efficiency_heatmap.png', dpi=300)
        plt.close()
    
    def plot_scaling_analysis(self):
        """Analyze performance scaling with problem size"""
        gemm_df = self.df[self.df['operation'] == 'GEMM'].copy()
        
        # Extract total operations
        gemm_df['total_ops'] = gemm_df['dimensions'].apply(
            lambda x: np.prod(eval(x) if isinstance(x, str) else x)
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for proc in gemm_df['processor'].unique():
            proc_data = gemm_df[gemm_df['processor'] == proc]
            ax.loglog(proc_data['total_ops'], proc_data['gflops'], 
                     marker='o', label=proc, linewidth=2)
        
        ax.set_xlabel('Total Operations')
        ax.set_ylabel('GFLOPS')
        ax.set_title('Performance Scaling Analysis')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_analysis.png', dpi=300)
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary report"""
        report_path = self.output_dir / 'benchmark_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("Apple Silicon Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("Overall Performance Statistics:\n")
            f.write("-" * 30 + "\n")
            
            for proc in self.df['processor'].unique():
                proc_data = self.df[self.df['processor'] == proc]
                f.write(f"\n{proc}:\n")
                f.write(f"  Average GFLOPS: {proc_data['gflops'].mean():.2f}\n")
                f.write(f"  Peak GFLOPS: {proc_data['gflops'].max():.2f}\n")
                f.write(f"  Average Latency: {proc_data['avg_time_ms'].mean():.2f} ms\n")
                
                if proc_data['power_watts'].sum() > 0:
                    avg_efficiency = proc_data['efficiency_gflops_per_watt'].mean()
                    f.write(f"  Average Efficiency: {avg_efficiency:.2f} GFLOPS/W\n")
            
            # Best configuration for each operation
            f.write("\n\nBest Processor by Operation:\n")
            f.write("-" * 30 + "\n")
            
            for op in self.df['operation'].unique():
                op_data = self.df[self.df['operation'] == op]
                best = op_data.loc[op_data['gflops'].idxmax()]
                f.write(f"{op}: {best['processor']} ({best['gflops']:.2f} GFLOPS)\n")
            
            # Memory bandwidth analysis
            f.write("\n\nMemory Bandwidth Utilization:\n")
            f.write("-" * 30 + "\n")
            
            for proc in self.df['processor'].unique():
                proc_data = self.df[self.df['processor'] == proc]
                avg_bw = proc_data['gb_per_sec'].mean()
                peak_bw = proc_data['gb_per_sec'].max()
                f.write(f"{proc}: Avg {avg_bw:.1f} GB/s, Peak {peak_bw:.1f} GB/s\n")
    
    def create_all_plots(self):
        """Generate all visualizations"""
        print("Generating performance plots...")
        
        self.plot_gemm_performance()
        print("  - GEMM performance plots created")
        
        self.plot_operation_comparison()
        print("  - Operation comparison plot created")
        
        self.plot_latency_distribution()
        print("  - Latency distribution plots created")
        
        self.plot_power_efficiency_heatmap()
        print("  - Power efficiency heatmap created")
        
        self.plot_scaling_analysis()
        print("  - Scaling analysis plot created")
        
        self.generate_summary_report()
        print("  - Summary report generated")
        
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Apple Silicon benchmark results')
    parser.add_argument('results_file', help='Path to benchmark results (JSON or CSV)')
    parser.add_argument('--output-dir', help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results_file)
    if args.output_dir:
        visualizer.output_dir = Path(args.output_dir)
        visualizer.output_dir.mkdir(exist_ok=True)
    
    visualizer.create_all_plots()


if __name__ == '__main__':
    main()