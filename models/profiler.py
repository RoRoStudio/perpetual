#!/usr/bin/env python3

# python -m models.run_training --quick-test --profile
"""
Direct profiling solution for model training.
This provides a more direct, robust approach to identify bottlenecks in training.

Usage:
1. Add this line to train_unified.py at the beginning of the main() function:
   profiler = TrainingProfiler("/mnt/p/perpetual/tmp")
   
2. Add this line at the end of main():
   profiler.save_results()
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import psutil
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class TrainingProfiler:
    """
    Simple, direct profiling for PyTorch training loops.
    """
    
    def __init__(self, output_dir="/mnt/p/perpetual/tmp"):
        """Initialize profiler with output directory."""
        self.output_dir = output_dir
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data structures
        self.timings = defaultdict(list)
        self.events = []
        self.memory_samples = []
        self.gpu_samples = []
        self.cpu_samples = []
        self.batch_times = []
        self.epoch_times = []
        
        # Track currently active operations
        self.active_operations = {}
        
        # Sample interval
        self.sample_interval = 5.0  # seconds
        self.last_sample_time = time.time()
        
        # Start monitoring thread
        self._start_sampling()
        
        print(f"🔍 Direct profiler initialized, results will be saved to {output_dir}")
    
    def _start_sampling(self):
        """Start periodic sampling of system metrics."""
        try:
            import threading
            self.keep_sampling = True
            self.sampling_thread = threading.Thread(target=self._sample_loop, daemon=True)
            self.sampling_thread.start()
        except Exception as e:
            print(f"Warning: Failed to start sampling thread: {e}")
    
    def _sample_loop(self):
        """Background thread that periodically samples system metrics."""
        while self.keep_sampling:
            try:
                self._record_memory_sample()
                time.sleep(0.5)  # Sample every 0.5 seconds
            except Exception as e:
                print(f"Error in sampling thread: {e}")
    
    def _record_memory_sample(self):
        """Record current memory usage."""
        now = time.time()
        
        # Only sample at the specified interval to keep file size reasonable
        if now - self.last_sample_time < self.sample_interval:
            return
            
        self.last_sample_time = now
        
        # CPU memory 
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        cpu_memory_used = memory_info.used / (1024 * 1024)  # MB
        
        # Process memory
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # GPU memory if available
        gpu_memory_used = 0
        gpu_utilization = 0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            try:
                # Try to get GPU utilization
                import subprocess
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
                )
                gpu_utilization = float(result.decode('utf-8').strip())
            except:
                pass
        
        # Store the sample
        sample = {
            'timestamp': now - self.start_time,  # seconds since start
            'cpu_percent': cpu_percent,
            'cpu_memory_mb': cpu_memory_used,
            'process_memory_mb': process_memory,
            'gpu_memory_mb': gpu_memory_used,
            'gpu_utilization': gpu_utilization
        }
        
        self.memory_samples.append(sample)
    
    def start_operation(self, name):
        """
        Start timing an operation.
        
        Args:
            name: Name of the operation
        """
        self.active_operations[name] = time.time()
        
        # Record event
        self.events.append({
            'type': 'start',
            'name': name,
            'time': time.time() - self.start_time
        })
    
    def end_operation(self, name):
        """
        End timing an operation and record its duration.
        
        Args:
            name: Name of the operation
        """
        if name in self.active_operations:
            start_time = self.active_operations[name]
            duration = time.time() - start_time
            
            # Store the timing
            self.timings[name].append(duration)
            
            # Remove from active operations
            del self.active_operations[name]
            
            # Record event
            self.events.append({
                'type': 'end',
                'name': name,
                'time': time.time() - self.start_time,
                'duration': duration
            })
            
            return duration
        return None
    
    def record_batch_time(self, duration):
        """Record time taken to process a batch."""
        self.batch_times.append(duration)
    
    def record_epoch_time(self, duration):
        """Record time taken to complete an epoch."""
        self.epoch_times.append(duration)
    
    def direct_profile(self, function_to_profile):
        """
        Simple decorator to profile a function.
        
        Usage:
        @profiler.direct_profile
        def function_to_profile():
            # function code
        """
        def wrapper(*args, **kwargs):
            operation_name = function_to_profile.__name__
            self.start_operation(operation_name)
            result = function_to_profile(*args, **kwargs)
            self.end_operation(operation_name)
            return result
        return wrapper
    
    def generate_summary(self):
        """Generate a concise, actionable summary of profiling results."""
        summary = {
            'overall': {
                'total_runtime': time.time() - self.start_time,
                'epochs': len(self.epoch_times),
                'batches': len(self.batch_times)
            },
            'operations': {},
            'memory': {
                'peak_process_memory_mb': 0,
                'peak_gpu_memory_mb': 0,
                'avg_cpu_percent': 0,
                'avg_gpu_utilization': 0
            },
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Calculate operation statistics
        total_profiled_time = 0
        
        for operation, times in self.timings.items():
            if not times:
                continue
                
            avg_time = np.mean(times)
            total_time = sum(times)
            total_profiled_time += total_time
            
            summary['operations'][operation] = {
                'avg_time': avg_time,
                'total_time': total_time,
                'call_count': len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        # Calculate memory statistics
        if self.memory_samples:
            process_memory = [s['process_memory_mb'] for s in self.memory_samples]
            gpu_memory = [s['gpu_memory_mb'] for s in self.memory_samples]
            cpu_percent = [s['cpu_percent'] for s in self.memory_samples]
            gpu_util = [s['gpu_utilization'] for s in self.memory_samples]
            
            summary['memory']['peak_process_memory_mb'] = max(process_memory)
            summary['memory']['peak_gpu_memory_mb'] = max(gpu_memory)
            summary['memory']['avg_cpu_percent'] = np.mean(cpu_percent)
            summary['memory']['avg_gpu_utilization'] = np.mean(gpu_util)
        
        # Calculate batch statistics
        if self.batch_times:
            summary['batch_stats'] = {
                'avg_batch_time': np.mean(self.batch_times),
                'min_batch_time': min(self.batch_times),
                'max_batch_time': max(self.batch_times),
                'samples_per_second': 1.0 / np.mean(self.batch_times)
            }
        
        # Calculate epoch statistics
        if self.epoch_times:
            summary['epoch_stats'] = {
                'avg_epoch_time': np.mean(self.epoch_times),
                'min_epoch_time': min(self.epoch_times),
                'max_epoch_time': max(self.epoch_times)
            }
        
        # Calculate operation percentages and identify bottlenecks
        sorted_ops = sorted(
            [(op, data['total_time']) for op, data in summary['operations'].items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Add percentage of total time
        for op, total_time in sorted_ops:
            if total_profiled_time > 0:
                pct = (total_time / total_profiled_time) * 100
                summary['operations'][op]['percent_of_profiled_time'] = pct
                
                # Identify bottlenecks (operations taking >10% of time)
                if pct > 10:
                    summary['bottlenecks'].append({
                        'operation': op,
                        'percent_of_time': pct,
                        'total_time': total_time
                    })
        
        # Make specific recommendations based on the bottlenecks
        if summary['bottlenecks']:
            for bottleneck in summary['bottlenecks']:
                op = bottleneck['operation']
                
                # Data loading bottleneck
                if 'dataloader' in op.lower() or 'data_transfer' in op.lower():
                    summary['recommendations'].append({
                        'bottleneck': op,
                        'recommendation': "Optimize data loading: increase num_workers, use pinned memory, try memory mapping, or use more efficient data format"
                    })
                
                # Forward pass bottleneck
                elif 'forward' in op.lower():
                    summary['recommendations'].append({
                        'bottleneck': op,
                        'recommendation': "Optimize model: simplify architecture, use faster operations, try a smaller model"
                    })
                
                # Backward pass bottleneck
                elif 'backward' in op.lower():
                    summary['recommendations'].append({
                        'bottleneck': op,
                        'recommendation': "Optimize backprop: use gradient checkpointing, try gradient accumulation with smaller batches"
                    })
                
                # Optimizer bottleneck
                elif 'optim' in op.lower():
                    summary['recommendations'].append({
                        'bottleneck': op,
                        'recommendation': "Use a more efficient optimizer or try adjusting learning rate and momentum"
                    })
                
                # Loss computation bottleneck
                elif 'loss' in op.lower():
                    summary['recommendations'].append({
                        'bottleneck': op,
                        'recommendation': "Simplify loss computation or use a more efficient implementation"
                    })
        
        # Check GPU utilization
        if summary['memory']['avg_gpu_utilization'] < 50 and torch.cuda.is_available():
            summary['recommendations'].append({
                'bottleneck': "Low GPU utilization",
                'recommendation': "CPU processing is likely bottlenecking your training. Optimize data loading operations."
            })
        
        # Add general recommendation for speeding up training
        summary['recommendations'].append({
            'bottleneck': "General Training Speed",
            'recommendation': "Try using mixed precision training with torch.cuda.amp, larger batch sizes, and gradient accumulation"
        })
        
        return summary
    
    def create_performance_timeline(self):
        """Create a DataFrame with timeline of operations for visualization."""
        if not self.events:
            return pd.DataFrame()
            
        # Create a sorted list of events
        sorted_events = sorted(self.events, key=lambda x: x['time'])
        
        # Convert to DataFrame
        timeline_df = pd.DataFrame(sorted_events)
        
        return timeline_df
    
    def create_memory_timeline(self):
        """Create a DataFrame with memory usage over time."""
        if not self.memory_samples:
            return pd.DataFrame()
            
        # Convert to DataFrame
        memory_df = pd.DataFrame(self.memory_samples)
        
        return memory_df
    
    def plot_timeline(self, save_path=None):
        """Plot timeline of operations."""
        timeline_df = self.create_performance_timeline()
        
        if timeline_df.empty:
            print("No timeline data available to plot")
            return
        
        # Get unique operation names
        operations = set()
        for event in self.events:
            if event['type'] == 'start':
                operations.add(event['name'])
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Set up colors
        colors = plt.cm.tab10.colors
        operation_colors = {op: colors[i % len(colors)] for i, op in enumerate(sorted(operations))}
        
        # Plot operations
        y_positions = {}
        y_pos = 0
        
        for op in sorted(operations):
            y_positions[op] = y_pos
            y_pos += 1
        
        # Find start and end times for each operation instance
        op_instances = []
        active_starts = {}
        
        for event in sorted(self.events, key=lambda x: x['time']):
            if event['type'] == 'start':
                active_starts[event['name']] = event['time']
            elif event['type'] == 'end' and event['name'] in active_starts:
                start_time = active_starts[event['name']]
                op_instances.append({
                    'name': event['name'],
                    'start': start_time,
                    'end': event['time'],
                    'duration': event['time'] - start_time
                })
                del active_starts[event['name']]
        
        # Plot each operation instance as a horizontal bar
        for op in op_instances:
            plt.barh(
                y_positions[op['name']],
                op['duration'],
                left=op['start'],
                height=0.8,
                color=operation_colors[op['name']],
                alpha=0.7
            )
        
        # Set y-ticks to operation names
        plt.yticks(list(y_positions.values()), list(y_positions.keys()))
        
        # Add labels and grid
        plt.xlabel('Time (seconds)')
        plt.title('Training Operation Timeline')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add legend
        for op, color in operation_colors.items():
            plt.plot([], [], color=color, label=op, linewidth=8)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved timeline plot to {save_path}")
        
        plt.close()
    
    def plot_memory_usage(self, save_path=None):
        """Plot memory usage over time."""
        memory_df = self.create_memory_timeline()
        
        if memory_df.empty:
            print("No memory data available to plot")
            return
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot CPU usage
        ax1.plot(memory_df['timestamp'], memory_df['cpu_percent'], 'b-')
        ax1.set_ylabel('CPU Utilization (%)')
        ax1.set_title('CPU Utilization')
        ax1.grid(True)
        
        # Plot memory usage
        ax2.plot(memory_df['timestamp'], memory_df['process_memory_mb'], 'r-', label='Process Memory')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Process Memory Usage')
        ax2.grid(True)
        
        # Plot GPU usage if available
        if torch.cuda.is_available():
            ax3.plot(memory_df['timestamp'], memory_df['gpu_memory_mb'], 'g-', label='GPU Memory')
            ax3.set_ylabel('GPU Memory (MB)')
            
            # Add GPU utilization on secondary y-axis
            ax3b = ax3.twinx()
            ax3b.plot(memory_df['timestamp'], memory_df['gpu_utilization'], 'm--', label='GPU Utilization')
            ax3b.set_ylabel('GPU Utilization (%)')
            
            # Create combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax3.set_title('GPU Usage')
            ax3.grid(True)
        
        # Set common x-label
        ax3.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved memory usage plot to {save_path}")
        
        plt.close()
    
    def plot_operation_breakdown(self, save_path=None):
        """Plot breakdown of time spent in different operations."""
        summary = self.generate_summary()
        
        if not summary['operations']:
            print("No operation data available to plot")
            return
        
        # Get operations sorted by total time
        ops = []
        times = []
        percentages = []
        
        for op, data in sorted(summary['operations'].items(), 
                              key=lambda x: x[1]['total_time'], 
                              reverse=True):
            ops.append(op)
            times.append(data['total_time'])
            percentages.append(data.get('percent_of_profiled_time', 0))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.barh(ops, times, color='skyblue')
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            if pct > 0:
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f"{pct:.1f}%", va='center')
        
        # Add labels and title
        plt.xlabel('Total Time (seconds)')
        plt.title('Time Spent in Training Operations')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved operation breakdown plot to {save_path}")
        
        plt.close()
    
    def save_results(self):
        """Save all profiling results."""
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"profile_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save summary as JSON
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary as text
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write("=== TRAINING PERFORMANCE SUMMARY ===\n\n")
            
            # Overall stats
            f.write(f"Total runtime: {summary['overall']['total_runtime']:.2f} seconds\n")
            f.write(f"Epochs: {summary['overall']['epochs']}\n")
            if 'batch_stats' in summary:
                f.write(f"Average batch time: {summary['batch_stats']['avg_batch_time']*1000:.2f} ms\n")
                f.write(f"Throughput: {summary['batch_stats']['samples_per_second']:.2f} samples/second\n")
            if 'epoch_stats' in summary:
                f.write(f"Average epoch time: {summary['epoch_stats']['avg_epoch_time']:.2f} seconds\n\n")
            
            # Memory stats
            f.write("=== MEMORY USAGE ===\n")
            f.write(f"Peak process memory: {summary['memory']['peak_process_memory_mb']:.2f} MB\n")
            f.write(f"Peak GPU memory: {summary['memory']['peak_gpu_memory_mb']:.2f} MB\n")
            f.write(f"Average CPU utilization: {summary['memory']['avg_cpu_percent']:.2f}%\n")
            f.write(f"Average GPU utilization: {summary['memory']['avg_gpu_utilization']:.2f}%\n\n")
            
            # Operation breakdown
            f.write("=== OPERATION BREAKDOWN ===\n")
            for op, data in sorted(summary['operations'].items(), 
                                key=lambda x: x[1].get('percent_of_profiled_time', 0), 
                                reverse=True):
                pct = data.get('percent_of_profiled_time', 0)
                f.write(f"{op}: {data['total_time']:.2f}s ({pct:.1f}%) - "
                       f"Avg: {data['avg_time']*1000:.2f}ms, "
                       f"Calls: {data['call_count']}\n")
            f.write("\n")
            
            # Bottlenecks
            f.write("=== BOTTLENECKS ===\n")
            for bottleneck in summary['bottlenecks']:
                f.write(f"{bottleneck['operation']}: {bottleneck['percent_of_time']:.1f}% "
                       f"of profiled time ({bottleneck['total_time']:.2f}s)\n")
            f.write("\n")
            
            # Recommendations
            f.write("=== RECOMMENDATIONS ===\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. For {rec['bottleneck']}: {rec['recommendation']}\n")
        
        # Save raw data
        with open(os.path.join(output_dir, "operations.json"), 'w') as f:
            json.dump(self.timings, f, indent=2)
        
        # Save timeline data
        with open(os.path.join(output_dir, "timeline.json"), 'w') as f:
            json.dump(self.events, f, indent=2)
        
        # Save memory data (summarized to avoid huge files)
        if self.memory_samples:
            # Sample every 10th point to reduce file size
            sampled_memory = self.memory_samples[::10]
            with open(os.path.join(output_dir, "memory.json"), 'w') as f:
                json.dump(sampled_memory, f, indent=2)
        
        # Generate plots
        self.plot_timeline(save_path=os.path.join(output_dir, "timeline.png"))
        self.plot_memory_usage(save_path=os.path.join(output_dir, "memory_usage.png"))
        self.plot_operation_breakdown(save_path=os.path.join(output_dir, "operation_breakdown.png"))
        
        print(f"✅ Profiling results saved to {output_dir}")
        
        # Print a short summary to console
        print("\n=== PERFORMANCE SUMMARY ===")
        if 'batch_stats' in summary:
            print(f"Average batch processing time: {summary['batch_stats']['avg_batch_time']*1000:.2f} ms")
            print(f"Training throughput: {summary['batch_stats']['samples_per_second']:.2f} samples/second")
        
        print("\nTop bottlenecks:")
        for bottleneck in summary['bottlenecks'][:3]:  # Top 3 bottlenecks
            print(f"- {bottleneck['operation']}: {bottleneck['percent_of_time']:.1f}% of time")
        
        print("\nKey recommendations:")
        for rec in summary['recommendations'][:3]:  # Top 3 recommendations
            print(f"- {rec['recommendation']}")
            
        print(f"\nDetailed profiling information available in {output_dir}/summary.txt")