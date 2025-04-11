#!/usr/bin/env python3

# python -m models.run_training --quick-test --profile

"""
High-precision profiling utilities for model training.
Provides detailed analysis of training bottlenecks across:
- Data loading time
- Data preprocessing
- Data transfer to GPU
- Forward pass time
- Backward pass time
- Optimizer step time
- GPU memory usage
- CPU memory usage
- GPU utilization
- Batch processing breakdown

Example usage:
```python
from models.profiler import TrainingProfiler

# Initialize profiler
profiler = TrainingProfiler(enabled=True, profile_gpu=True)

# Profile training loop
with profiler.profile_region("epoch"):
    for batch_idx, (features, targets) in enumerate(train_loader):
        with profiler.profile_region("data_transfer"):
            # Move data to GPU
            features = features.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
        
        with profiler.profile_region("forward"):
            outputs = model(features)
            
        with profiler.profile_region("loss_computation"):
            loss = criterion(outputs, targets)
            
        with profiler.profile_region("backward"):
            loss.backward()
            
        with profiler.profile_region("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()
            
        # Log batch profiling data
        profiler.step()

# Generate report
profiler.print_summary()
profiler.export_chrome_trace("/tmp/train_trace.json")
profiler.plot_timeline("training_timeline.png")
```
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import torch
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union
import psutil
import json
import warnings
from pathlib import Path
import gc
import threading
from datetime import datetime
import atexit
import signal

class TrainingProfiler:
    """
    Comprehensive profiling tool for PyTorch training loops.
    Tracks detailed metrics on CPU/GPU usage, timing, and bottlenecks.
    """
    
    def __init__(
        self, 
        enabled: bool = True, 
        profile_gpu: bool = True,
        profile_memory: bool = True,
        trace_cuda: bool = False,
        warmup_steps: int = 5,
        logging_steps: int = 10, 
        detailed_gpu: bool = False,
        profile_dir: str = "/tmp/training_profile"
    ):
        """
        Initialize profiler with options.
        
        Args:
            enabled: Whether profiling is active
            profile_gpu: Enable GPU profiling
            profile_memory: Track memory usage
            trace_cuda: Capture CUDA kernel traces (can be slow)
            warmup_steps: Number of steps to skip before profiling
            logging_steps: How often to log detailed metrics
            detailed_gpu: Capture detailed GPU metrics (op-level)
            profile_dir: Directory to save profiling data
        """
        self.enabled = enabled
        self.profile_gpu = profile_gpu and torch.cuda.is_available()
        self.profile_memory = profile_memory
        self.trace_cuda = trace_cuda and torch.cuda.is_available()
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.detailed_gpu = detailed_gpu and torch.cuda.is_available()
        self.profile_dir = profile_dir
        
        # Initialize lock for thread safety
        self._lock = threading.Lock()
        
        # Create profile directory
        if self.enabled:
            os.makedirs(profile_dir, exist_ok=True)
        
        # Initialize storage
        self.reset()
        
        # Setup PyTorch profiler if needed
        self._setup_pytorch_profiler()
        
        # Start background monitoring thread
        self._setup_background_monitoring()
        
        # Register exit handlers
        atexit.register(self.save_summary)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🔍 Training profiler initialized {'(enabled)' if enabled else '(disabled)'}")
        if self.enabled:
            print(f"   - GPU profiling: {'✅' if self.profile_gpu else '❌'}")
            print(f"   - Memory tracking: {'✅' if self.profile_memory else '❌'}")
            print(f"   - CUDA tracing: {'✅' if self.trace_cuda else '❌'}")
            print(f"   - Profile directory: {self.profile_dir}")
    
    def reset(self):
        """Reset all profiling data."""
        if not self.enabled:
            return
            
        self.step_counter = 0
        self.region_times = defaultdict(list)
        self.active_regions = {}
        self.memory_usage = {
            'cpu': [],
            'gpu': []
        }
        self.gpu_utilization = []
        self.timestamps = []
        self.batch_sizes = []
        self.batch_process_times = []
        self.stall_times = []  # Time between batch completion and next batch start
        self.curr_stall_start = None
        
        # Timeline data
        self.timeline = []
        
        # Set baseline memory usage
        self._capture_memory_usage()
        
        # Keep active timers for nested regions
        self.active_timers = {}
    
    def _setup_pytorch_profiler(self):
        """Initialize PyTorch profiler if needed."""
        if not self.enabled or not self.profile_gpu:
            self.torch_profiler = None
            return
            
        try:
            from torch.profiler import profile, record_function
            from torch.profiler import ProfilerActivity
            
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            # We'll create the actual profiler in the step() method
            self.profiler_activities = activities
            self.record_function = record_function
            self.torch_profiler_cls = profile
            self.active_torch_profiler = None
            
            # Set some profiler config
            self.profiler_schedule = torch.profiler.schedule(
                wait=self.warmup_steps,
                warmup=2,
                active=1,
                repeat=1
            )
            
            print("✅ PyTorch profiler configured successfully")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize PyTorch profiler: {e}")
            self.torch_profiler = None
    
    def _setup_background_monitoring(self):
        """Setup background thread for monitoring system resources."""
        if not self.enabled or not self.profile_memory:
            self.monitoring_thread = None
            return
            
        self.keep_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitor_resources(self):
        """Background thread to monitor system resources."""
        try:
            while self.keep_monitoring:
                # Get CPU and GPU memory
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                
                gpu_mem = 0
                gpu_util = 0
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    
                    # Try to get GPU utilization if nvidia-smi is available
                    try:
                        import subprocess
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
                        )
                        gpu_util = float(result.decode('utf-8').strip())
                    except:
                        pass
                
                # Store the data with timestamp
                timestamp = time.time()
                with self._lock:
                    self.memory_usage['cpu'].append(cpu_mem)
                    self.memory_usage['gpu'].append(gpu_mem)
                    self.gpu_utilization.append(gpu_util)
                    self.timestamps.append(timestamp)
                
                # Sleep for a short period
                time.sleep(0.2)
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
    
    def _capture_memory_usage(self):
        """Capture current memory usage."""
        if not self.enabled or not self.profile_memory:
            return
            
        # CPU memory
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # GPU memory
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            
        return cpu_mem, gpu_mem
    
    @contextmanager
    def profile_region(self, region_name: str):
        """
        Context manager to profile a specific region of code.
        
        Args:
            region_name: Name of the region to profile
        """
        if not self.enabled:
            yield
            return
            
        # Record start time
        start_time = time.time()
        self.active_regions[region_name] = start_time
        
        # Create PyTorch profiler record function if needed
        record_context = (
            self.record_function(region_name) 
            if self.profile_gpu and hasattr(self, 'record_function') 
            else nullcontext()
        )
        
        # Add to timeline
        self.timeline.append({
            'event': 'region_start',
            'region': region_name,
            'time': start_time,
            'step': self.step_counter
        })
        
        # Enter the region
        with record_context:
            try:
                # Capture memory before
                if self.profile_memory:
                    mem_before = self._capture_memory_usage()
                
                # Execute the region
                yield
                
            finally:
                # Record end time
                end_time = time.time()
                duration = end_time - start_time
                
                # Capture memory after
                if self.profile_memory:
                    mem_after = self._capture_memory_usage()
                    mem_diff = (
                        mem_after[0] - mem_before[0],  # CPU diff
                        mem_after[1] - mem_before[1]   # GPU diff
                    )
                else:
                    mem_diff = (0, 0)
                
                # Store timing data
                self.region_times[region_name].append({
                    'duration': duration,
                    'step': self.step_counter,
                    'cpu_mem_diff': mem_diff[0],
                    'gpu_mem_diff': mem_diff[1],
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                # Add to timeline
                self.timeline.append({
                    'event': 'region_end',
                    'region': region_name,
                    'time': end_time,
                    'duration': duration,
                    'step': self.step_counter,
                    'cpu_mem_diff': mem_diff[0],
                    'gpu_mem_diff': mem_diff[1]
                })
                
                # Clear active region
                self.active_regions.pop(region_name, None)
    
    def step(self, batch_size: Optional[int] = None):
        """
        Call after each training step to collect profiling data.
        
        Args:
            batch_size: Size of the current batch
        """
        if not self.enabled:
            return
            
        # Record stall time if we have a previous batch completion
        if self.curr_stall_start is not None:
            stall_time = time.time() - self.curr_stall_start
            self.stall_times.append(stall_time)
            
            # Add to timeline
            self.timeline.append({
                'event': 'stall',
                'time': self.curr_stall_start,
                'duration': stall_time,
                'step': self.step_counter
            })
        
        # Process PyTorch profiler data
        if (self.profile_gpu and 
            hasattr(self, 'torch_profiler_cls') and 
            self.step_counter >= self.warmup_steps and 
            self.step_counter % self.logging_steps == 0):
            
            # Start a new profiler if needed
            if self.active_torch_profiler is None:
                self.active_torch_profiler = self.torch_profiler_cls(
                    activities=self.profiler_activities,
                    schedule=self.profiler_schedule,
                    on_trace_ready=self._save_chrome_trace,
                    with_stack=True,
                    with_modules=True,
                    with_flops=True,
                    record_shapes=True,
                    profile_memory=True
                )
                self.active_torch_profiler.__enter__()
            
            # Step the profiler
            if self.active_torch_profiler is not None:
                self.active_torch_profiler.step()
                
                # If we've completed the schedule, exit and reset
                if not self.profiler_schedule(self.step_counter - self.warmup_steps):
                    self.active_torch_profiler.__exit__(None, None, None)
                    self.active_torch_profiler = None
        
        # Calculate total time spent in all regions
        total_region_time = 0
        for region, start_time in list(self.active_regions.items()):
            current_time = time.time()
            duration = current_time - start_time
            total_region_time += duration
        
        # Record batch size
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
        
        # Record batch process time
        if self.step_counter > 0:  # Skip first step as it may include compilation time
            process_time = sum(
                region_data[-1]['duration'] 
                for region_data in self.region_times.values() 
                if len(region_data) > 0
            )
            self.batch_process_times.append(process_time)
        
        # Set stall start time for next batch
        self.curr_stall_start = time.time()
        
        # Increment step counter
        self.step_counter += 1
        
        # Log profiling data periodically
        if self.step_counter % self.logging_steps == 0:
            self._log_periodic_data()
    
    def _log_periodic_data(self):
        """Log periodic profiling data."""
        if not self.enabled:
            return
            
        # Get timing data for each region
        timing_data = {}
        for region, times in self.region_times.items():
            if times:
                recent_times = [t['duration'] for t in times[-self.logging_steps:]]
                if recent_times:
                    avg_time = sum(recent_times) / len(recent_times)
                    timing_data[region] = avg_time
        
        # Calculate throughput
        if self.batch_sizes and self.batch_process_times:
            recent_batch_sizes = self.batch_sizes[-self.logging_steps:]
            recent_process_times = self.batch_process_times[-self.logging_steps:]
            if recent_batch_sizes and recent_process_times:
                total_samples = sum(recent_batch_sizes)
                total_time = sum(recent_process_times)
                throughput = total_samples / total_time if total_time > 0 else 0
                timing_data['throughput_samples_per_sec'] = throughput
        
        # Get average stall time
        if self.stall_times:
            recent_stall_times = self.stall_times[-self.logging_steps:]
            if recent_stall_times:
                avg_stall = sum(recent_stall_times) / len(recent_stall_times)
                timing_data['dataloader_stall_time'] = avg_stall
        
        # Get memory usage
        if self.profile_memory:
            cpu_mem, gpu_mem = self._capture_memory_usage()
            timing_data['cpu_memory_mb'] = cpu_mem
            timing_data['gpu_memory_mb'] = gpu_mem
        
        # Log performance data
        print(f"\n--- Profiling Step {self.step_counter} ---")
        print(f"Region Times (ms):")
        for region, time_ms in sorted(timing_data.items(), key=lambda x: x[1], reverse=True):
            if region.startswith('throughput') or region.startswith('dataloader') or region.startswith('cpu_') or region.startswith('gpu_'):
                continue
            print(f"  {region}: {time_ms*1000:.2f}")
        
        if 'throughput_samples_per_sec' in timing_data:
            print(f"Throughput: {timing_data['throughput_samples_per_sec']:.2f} samples/sec")
        
        if 'dataloader_stall_time' in timing_data:
            print(f"DataLoader Stall: {timing_data['dataloader_stall_time']*1000:.2f} ms")
        
        if 'cpu_memory_mb' in timing_data and 'gpu_memory_mb' in timing_data:
            print(f"Memory: CPU {timing_data['cpu_memory_mb']:.2f} MB, GPU {timing_data['gpu_memory_mb']:.2f} MB")
    
    def _save_chrome_trace(self, prof):
        """Save Chrome trace file from PyTorch profiler."""
        if not self.enabled or not self.profile_gpu:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = os.path.join(self.profile_dir, f"torch_trace_{timestamp}_{self.step_counter}.json")
            prof.export_chrome_trace(trace_path)
            print(f"💾 Saved PyTorch trace to {trace_path}")
        except Exception as e:
            warnings.warn(f"Failed to save Chrome trace: {e}")
    
    def export_chrome_trace(self, path: str):
        """
        Export timeline as Chrome trace format.
        
        Args:
            path: Path to save trace file
        """
        if not self.enabled:
            return
            
        try:
            # Convert timeline to Chrome trace format
            trace_events = []
            
            # Process region events
            for event in self.timeline:
                if event['event'] == 'region_start':
                    trace_events.append({
                        "name": event['region'],
                        "ph": "B",  # Begin
                        "ts": (event['time'] * 1000000),  # Convert to microseconds
                        "pid": 1,
                        "tid": 1
                    })
                elif event['event'] == 'region_end':
                    trace_events.append({
                        "name": event['region'],
                        "ph": "E",  # End
                        "ts": (event['time'] * 1000000),  # Convert to microseconds
                        "pid": 1,
                        "tid": 1
                    })
                elif event['event'] == 'stall':
                    trace_events.append({
                        "name": "DataLoader Stall",
                        "ph": "X",  # Complete
                        "ts": (event['time'] * 1000000),  # Convert to microseconds
                        "dur": (event['duration'] * 1000000),  # Convert to microseconds
                        "pid": 1,
                        "tid": 2
                    })
            
            # Write trace file
            with open(path, 'w') as f:
                json.dump({
                    "traceEvents": trace_events
                }, f)
                
            print(f"💾 Exported Chrome trace to {path}")
            print(f"    View at chrome://tracing/ or https://ui.perfetto.dev/")
            
        except Exception as e:
            warnings.warn(f"Failed to export Chrome trace: {e}")
    
    def print_summary(self):
        """Print summary of profiling results."""
        if not self.enabled:
            return
            
        print("\n" + "="*50)
        print("PROFILING SUMMARY")
        print("="*50)
        
        # Calculate average time for each region
        region_avg_times = {}
        for region, times in self.region_times.items():
            if times:
                # Skip first few steps which may include compilation
                valid_times = times[self.warmup_steps:]
                if valid_times:
                    avg_time = sum(t['duration'] for t in valid_times) / len(valid_times)
                    region_avg_times[region] = avg_time
        
        # Print region timings
        print("\nRegion Times (sorted by average duration):")
        total_time = sum(region_avg_times.values())
        for region, avg_time in sorted(region_avg_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (avg_time / total_time * 100) if total_time > 0 else 0
            print(f"  {region}: {avg_time*1000:.2f} ms ({percentage:.1f}%)")
        
        # Print throughput statistics
        if self.batch_sizes and self.batch_process_times:
            # Skip first few steps
            valid_batch_sizes = self.batch_sizes[self.warmup_steps:]
            valid_process_times = self.batch_process_times[self.warmup_steps:]
            
            if valid_batch_sizes and valid_process_times:
                total_samples = sum(valid_batch_sizes)
                total_time = sum(valid_process_times)
                avg_throughput = total_samples / total_time if total_time > 0 else 0
                
                print(f"\nThroughput:")
                print(f"  Average: {avg_throughput:.2f} samples/sec")
                print(f"  Total Samples: {total_samples}")
                print(f"  Total Process Time: {total_time:.2f} sec")
        
        # Print dataloader stall statistics
        if self.stall_times:
            valid_stall_times = self.stall_times[self.warmup_steps:]
            if valid_stall_times:
                avg_stall = sum(valid_stall_times) / len(valid_stall_times)
                max_stall = max(valid_stall_times)
                min_stall = min(valid_stall_times)
                
                print(f"\nDataLoader Stall Times:")
                print(f"  Average: {avg_stall*1000:.2f} ms")
                print(f"  Maximum: {max_stall*1000:.2f} ms")
                print(f"  Minimum: {min_stall*1000:.2f} ms")
        
        # Print memory statistics
        if self.profile_memory and self.memory_usage['cpu'] and self.memory_usage['gpu']:
            cpu_max = max(self.memory_usage['cpu'])
            gpu_max = max(self.memory_usage['gpu'])
            
            print(f"\nMemory Usage:")
            print(f"  CPU Peak: {cpu_max:.2f} MB")
            print(f"  GPU Peak: {gpu_max:.2f} MB")
        
        print("\nBottleneck Analysis:")
        # Detect if GPU is underutilized
        if self.profile_memory and self.gpu_utilization:
            avg_gpu_util = sum(self.gpu_utilization) / len(self.gpu_utilization)
            if avg_gpu_util < 50:
                print(f"  ⚠️ Low GPU utilization ({avg_gpu_util:.1f}%) - CPU or data loading may be the bottleneck")
        
        # Check if dataloader is bottleneck
        if self.stall_times:
            valid_stall_times = self.stall_times[self.warmup_steps:]
            valid_process_times = self.batch_process_times[self.warmup_steps:] if self.batch_process_times else []
            
            if valid_stall_times and valid_process_times:
                avg_stall = sum(valid_stall_times) / len(valid_stall_times)
                avg_process = sum(valid_process_times) / len(valid_process_times)
                
                if avg_stall > 0.5 * avg_process:
                    print(f"  ⚠️ DataLoader stalls ({avg_stall*1000:.1f} ms) are significant compared to processing time ({avg_process*1000:.1f} ms)")
                    print(f"     Consider increasing num_workers or using more efficient data loading")
        
        # Check if CPU preprocessing is bottleneck
        if 'data_transfer' in region_avg_times and 'forward' in region_avg_times:
            data_transfer_time = region_avg_times['data_transfer']
            forward_time = region_avg_times['forward']
            
            if data_transfer_time > 0.3 * forward_time:
                print(f"  ⚠️ Data transfer time ({data_transfer_time*1000:.1f} ms) is significant compared to forward pass ({forward_time*1000:.1f} ms)")
                print(f"     Consider using faster data transfer methods (pinned memory, GPU preprocessing)")
        
        # Print recommendations
        print("\nRecommendations:")
        
        if self.stall_times and sum(self.stall_times) / len(self.stall_times) > 0.01:
            print("  • Optimize data loading:")
            print("    - Increase num_workers")
            print("    - Use memory mapping")
            print("    - Prefetch more batches")
            print("    - Use more efficient data format")
        
        if 'data_transfer' in region_avg_times and region_avg_times.get('data_transfer', 0) > 0.01:
            print("  • Optimize data transfers:")
            print("    - Use pinned memory (pin_memory=True)")
            print("    - Pre-process data on GPU")
            print("    - Use non_blocking transfers")
            print("    - Use mixed precision to reduce transfer size")
            
        if 'backward' in region_avg_times and 'forward' in region_avg_times and region_avg_times['backward'] > 2 * region_avg_times['forward']:
            print("  • Optimize backward pass:")
            print("    - Use gradient accumulation")
            print("    - Use gradient checkpointing")
            print("    - Reduce model size or batch size")
            print("    - Use faster optimizer")
            
        if 'optimizer_step' in region_avg_times and region_avg_times.get('optimizer_step', 0) > 0.01:
            print("  • Optimize optimizer:")
            print("    - Use a more efficient optimizer")
            print("    - Apply gradient clipping before optimizer step")
            print("    - Use parameter groups with different learning rates")
            
        print("="*50)
    
    def save_summary(self, path: Optional[str] = None):
        """
        Save profiling summary to files.
        
        Args:
            path: Directory to save summary files
        """
        if not self.enabled:
            return
            
        try:
            # Use provided path or default
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.profile_dir, f"profile_summary_{timestamp}")
            
            os.makedirs(path, exist_ok=True)
            
            # Save timeline
            timeline_path = os.path.join(path, "timeline.json")
            with open(timeline_path, 'w') as f:
                json.dump(self.timeline, f)
            
            # Save region times
            region_times_path = os.path.join(path, "region_times.json")
            with open(region_times_path, 'w') as f:
                json.dump({k: v for k, v in self.region_times.items()}, f)
            
            # Save memory usage
            if self.profile_memory:
                memory_path = os.path.join(path, "memory_usage.json")
                with open(memory_path, 'w') as f:
                    json.dump({
                        'timestamps': self.timestamps,
                        'cpu': self.memory_usage['cpu'],
                        'gpu': self.memory_usage['gpu'],
                        'gpu_util': self.gpu_utilization
                    }, f)
            
            # Create and save visualizations
            self.plot_timeline(os.path.join(path, "timeline.png"))
            self.plot_memory_usage(os.path.join(path, "memory_usage.png"))
            
            print(f"💾 Saved profiling summary to {path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save profiling summary: {e}")
    
    def plot_timeline(self, path: str):
        """
        Create and save timeline visualization.
        
        Args:
            path: Path to save the plot
        """
        if not self.enabled or not self.timeline:
            return
            
        try:
            # Process timeline data
            regions = set(event['region'] for event in self.timeline 
                         if 'region' in event and event['event'] in ['region_start', 'region_end'])
            
            region_data = defaultdict(list)
            for i in range(len(self.timeline)):
                event = self.timeline[i]
                
                if event['event'] == 'region_start':
                    region = event['region']
                    start_time = event['time']
                    
                    # Find matching end event
                    for j in range(i+1, len(self.timeline)):
                        if (self.timeline[j]['event'] == 'region_end' and 
                            self.timeline[j]['region'] == region):
                            end_time = self.timeline[j]['time']
                            duration = end_time - start_time
                            
                            region_data[region].append({
                                'start': start_time,
                                'end': end_time,
                                'duration': duration,
                                'step': event['step']
                            })
                            break
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            
            # Set up colors
            colors = plt.cm.tab10.colors
            region_colors = {region: colors[i % len(colors)] 
                            for i, region in enumerate(sorted(regions))}
            
            # Plot each region
            for i, (region, events) in enumerate(sorted(region_data.items())):
                for event in events:
                    plt.barh(
                        i, 
                        event['duration'], 
                        left=event['start'] - self.timeline[0]['time'],  # Normalize to start time
                        color=region_colors[region],
                        alpha=0.7
                    )
            
            # Plot stall times
            stall_events = [event for event in self.timeline if event['event'] == 'stall']
            if stall_events:
                for event in stall_events:
                    plt.barh(
                        len(regions), 
                        event['duration'], 
                        left=event['time'] - self.timeline[0]['time'],  # Normalize to start time
                        color='red',
                        alpha=0.5
                    )
                
                # Add stall label
                regions = list(regions) + ['DataLoader Stall']
            
            # Set labels and title
            plt.yticks(range(len(regions)), regions)
            plt.xlabel('Time (seconds)')
            plt.title('Training Timeline')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add legend
            for region, color in region_colors.items():
                plt.plot([], [], color=color, label=region, linewidth=8)
            if stall_events:
                plt.plot([], [], color='red', label='DataLoader Stall', linewidth=8, alpha=0.5)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Failed to create timeline plot: {e}")
    
    def plot_memory_usage(self, path: str):
        """
        Create and save memory usage visualization.
        
        Args:
            path: Path to save the plot
        """
        if not self.enabled or not self.profile_memory:
            return
            
        try:
            # Check if we have memory data
            if not self.timestamps or not self.memory_usage['cpu']:
                return
                
            # Normalize timestamps to start at 0
            start_time = self.timestamps[0]
            x = [t - start_time for t in self.timestamps]
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Plot CPU memory
            ax1.plot(x, self.memory_usage['cpu'], 'b-', label='CPU Memory')
            ax1.set_ylabel('Memory (MB)')
            ax1.set_title('CPU Memory Usage')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot GPU memory if available
            if torch.cuda.is_available() and self.memory_usage['gpu']:
                ax2.plot(x, self.memory_usage['gpu'], 'r-', label='GPU Memory')
                
                # Plot GPU utilization if available
                if self.gpu_utilization:
                    ax3 = ax2.twinx()
                    ax3.plot(x, self.gpu_utilization, 'g-', label='GPU Utilization')
                    ax3.set_ylabel('Utilization (%)')
                    ax3.legend(loc='upper right')
                
                ax2.set_ylabel('Memory (MB)')
                ax2.set_title('GPU Memory Usage and Utilization')
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Set common labels
            ax2.set_xlabel('Time (seconds)')
            
            plt.tight_layout()
            plt.savefig(path, dpi=100)
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Failed to create memory usage plot: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle process termination signal."""
        self.save_summary()
        sys.exit(0)
    
    def __del__(self):
        """Clean up resources when profiler is deleted."""
        if hasattr(self, 'keep_monitoring'):
            self.keep_monitoring = False
        
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)


@contextmanager
def nullcontext():
    """Simple context manager that does nothing."""
    yield


# For compatibility with PyTorch profiler
class ProfilerContext:
    """Context manager for profiling a section of code."""
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        return self.profiler.profile_region(self.name).__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.profiler.profile_region(self.name).__exit__(exc_type, exc_val, exc_tb)


def apply_profiler_to_train_unified():
    """
    Integrate profiler with train_unified.py
    Example usage in train_unified.py:
    
    ```python
    from models.profiler import apply_profiler_to_train_unified
    
    # In main function:
    profiler = apply_profiler_to_train_unified() if args.profile else None
    
    # Pass profiler to train_unified_model:
    train_unified_model(..., profiler=profiler)
    ```
    """
    profiler = TrainingProfiler(
        enabled=True,
        profile_gpu=True,
        profile_memory=True,
        trace_cuda=False,
        warmup_steps=5,
        logging_steps=10
    )
    
    # Monkey patch the train_optimized function
    import models.train_unified
    original_train_optimized = models.train_unified.train_optimized
    
    def patched_train_optimized(model, train_loader, val_loader, optimizer, scheduler, 
                               scaler, num_epochs=20, device="cuda", patience=5, 
                               gradient_accumulation=1, model_save_path="/mnt/p/perpetual/models/checkpoints",
                               experiment_name=None, debug=False, use_wandb=True):
        """Patched version of train_optimized with profiling."""
        print("🔍 Using patched train_optimized with profiling")
        
        # Initialize metrics
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Ensure model save directory exists
        os.makedirs(model_save_path, exist_ok=True)
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            checkpoint_base = f"{model_save_path}/{experiment_name}_{timestamp}"
        else:
            checkpoint_base = f"{model_save_path}/model_{timestamp}"
        
        # Create a history dictionary to track metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_direction_acc': [],
            'val_direction_acc': [],
            'learning_rate': []
        }
        
        # Set torch.cuda.amp precision
        torch.set_float32_matmul_precision('high')
        
        # Define loss functions
        direction_criterion = models.train_unified.nn.CrossEntropyLoss()
        regression_criterion = models.train_unified.nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            with profiler.profile_region(f"epoch_{epoch}"):
                epoch_start_time = time.time()
                
                # ===== TRAINING =====
                with profiler.profile_region("train_mode"):
                    model.train()
                
                train_losses = []
                train_direction_correct = 0
                train_direction_total = 0
                train_return_mse_sum = 0
                
                with profiler.profile_region("optimizer_zero_grad"):
                    optimizer.zero_grad()  # Zero gradients at start of epoch
                
                # Use tqdm with a higher update frequency for more responsive UI
                for batch_idx, (features, targets) in enumerate(models.train_unified.tqdm(
                        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", miniters=10)):
                    
                    batch_size = features.size(0)  # Get batch size for profiler
                    
                    with profiler.profile_region("data_transfer"):
                        # Move to device and ensure data is clean - use non_blocking for async transfer
                        features = features.to(device, non_blocking=True)
                        features = models.train_unified.torch.nan_to_num(features)
                        
                        # Move all target tensors to device with non_blocking
                        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                    
                    # Get instrument IDs for asset-aware models
                    asset_ids = targets.get('instrument_id')
                    
                    # Mixed precision forward pass
                    with profiler.profile_region("forward"):
                        with models.train_unified.autocast():
                            outputs = model(features, targets['funding_rate'], asset_ids)
                    
                    with profiler.profile_region("loss_computation"):
                        with models.train_unified.autocast():
                            # Compute losses
                            direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                            return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                            risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                            
                            # Combined loss with weighting
                            loss = direction_loss + return_loss + 0.5 * risk_loss
                            
                            # Scale loss by gradient accumulation steps
                            loss = loss / gradient_accumulation
                    
                    # Mixed precision backward pass
                    with profiler.profile_region("backward"):
                        scaler.scale(loss).backward()
                    
                    # Only update weights after accumulating gradients
                    if (batch_idx + 1) % gradient_accumulation == 0:
                        with profiler.profile_region("optimizer_step"):
                            # Unscale gradients for clipping
                            scaler.unscale_(optimizer)
                            
                            # Clip gradients to prevent explosion
                            models.train_unified.torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # Update weights and zero gradients
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster than setting to zero
                        
                        with profiler.profile_region("scheduler_step"):
                            # Update learning rate each batch with OneCycleLR
                            scheduler.step()
                    
                    # Record metrics
                    train_losses.append(loss.item() * gradient_accumulation)  # Rescale loss back
                    
                    # Calculate accuracy
                    pred_direction = models.train_unified.torch.argmax(outputs['direction_logits'], dim=1)
                    train_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                    train_direction_total += targets['direction_class'].size(0)
                    
                    # Calculate MSE for returns
                    train_return_mse_sum += return_loss.item() * gradient_accumulation * targets['next_return_1bar'].size(0)
                    
                    # Step the profiler
                    profiler.step(batch_size=batch_size)
                    
                    # Log batch metrics to W&B (but not too frequently)
                    if use_wandb and batch_idx % 20 == 0:
                        # Calculate GPU memory metrics
                        if models.train_unified.torch.cuda.is_available():
                            gpu_memory_allocated = models.train_unified.torch.cuda.memory_allocated(0) / (1024**3)
                            gpu_memory_reserved = models.train_unified.torch.cuda.memory_reserved(0) / (1024**3)
                            gpu_utilization = 0  # Will be tracked by W&B system metrics
                            
                            # Log memory usage
                            models.train_unified.wandb.log({
                                'gpu_memory_allocated_GB': gpu_memory_allocated,
                                'gpu_memory_reserved_GB': gpu_memory_reserved,
                                'iteration': batch_idx + epoch * len(train_loader)
                            })
                            
                        models.train_unified.wandb.log({
                            'batch_loss': loss.item() * gradient_accumulation,
                            'batch_direction_loss': direction_loss.item(),
                            'batch_return_loss': return_loss.item(),
                            'batch_risk_loss': risk_loss.item(),
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'batch': batch_idx + epoch * len(train_loader)
                        })
                
                # Calculate epoch metrics
                train_loss = models.train_unified.np.mean(train_losses)
                train_direction_acc = train_direction_correct / train_direction_total if train_direction_total > 0 else 0
                train_return_mse = train_return_mse_sum / train_direction_total if train_direction_total > 0 else 0
                
                # Force garbage collection
                with profiler.profile_region("garbage_collection"):
                    models.train_unified.gc.collect()
                    if models.train_unified.torch.cuda.is_available():
                        models.train_unified.torch.cuda.empty_cache()
                
                # ===== VALIDATION =====
                with profiler.profile_region("validation"):
                    with profiler.profile_region("eval_mode"):
                        model.eval()
                    
                    val_losses = []
                    val_direction_correct = 0
                    val_direction_total = 0
                    val_return_mse_sum = 0
                    
                    with models.train_unified.torch.no_grad():
                        for val_batch_idx, (features, targets) in enumerate(models.train_unified.tqdm(
                                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                            
                            with profiler.profile_region("validation_data_transfer"):
                                # Move to device
                                features = features.to(device, non_blocking=True)
                                features = models.train_unified.torch.nan_to_num(features)
                                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                                asset_ids = targets.get('instrument_id')
                            
                            # Mixed precision inference
                            with profiler.profile_region("validation_forward"):
                                with models.train_unified.autocast():
                                    outputs = model(features, targets['funding_rate'], asset_ids)
                            
                            with profiler.profile_region("validation_loss"):
                                with models.train_unified.autocast():
                                    # Compute losses
                                    direction_loss = direction_criterion(outputs['direction_logits'], targets['direction_class'].squeeze())
                                    return_loss = regression_criterion(outputs['expected_return'], targets['next_return_1bar'])
                                    risk_loss = regression_criterion(outputs['expected_risk'], targets['next_volatility'])
                                    loss = direction_loss + return_loss + 0.5 * risk_loss
                            
                            # Record metrics
                            val_losses.append(loss.item())
                            
                            # Calculate accuracy
                            pred_direction = models.train_unified.torch.argmax(outputs['direction_logits'], dim=1)
                            val_direction_correct += (pred_direction == targets['direction_class'].squeeze()).sum().item()
                            val_direction_total += targets['direction_class'].size(0)
                            
                            # Calculate MSE
                            val_return_mse_sum += return_loss.item() * targets['next_return_1bar'].size(0)
                            
                            # Step the profiler for validation too
                            if val_batch_idx % 10 == 0:  # Less frequent updates for validation
                                profiler.step(batch_size=features.size(0))
                
                # Calculate epoch metrics
                val_loss = models.train_unified.np.mean(val_losses)
                val_direction_acc = val_direction_correct / val_direction_total if val_direction_total > 0 else 0
                val_return_mse = val_return_mse_sum / val_direction_total if val_direction_total > 0 else 0
                
                # Calculate epoch duration
                epoch_duration = time.time() - epoch_start_time
                
                # Update history
                current_lr = optimizer.param_groups[0]['lr']
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_direction_acc'].append(train_direction_acc)
                history['val_direction_acc'].append(val_direction_acc)
                history['learning_rate'].append(current_lr)
                
                # Print epoch results
                print(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Time: {epoch_duration:.2f}s - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f} - "
                    f"Train Acc: {train_direction_acc:.4f} - "
                    f"Val Acc: {val_direction_acc:.4f} - "
                    f"LR: {current_lr:.6f}")
                
                # Log to W&B
                if use_wandb:
                    models.train_unified.wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_direction_acc': train_direction_acc,
                        'val_direction_acc': val_direction_acc,
                        'train_return_mse': train_return_mse,
                        'val_return_mse': val_return_mse,
                        'learning_rate': current_lr,
                        'epoch_duration': epoch_duration
                    })
                
                # Save if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    with profiler.profile_region("save_model"):
                        checkpoint_path = f"{checkpoint_base}_best.pt"
                        models.train_unified.torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'history': history
                        }, checkpoint_path)
                    
                    print(f"✅ Model improved! Saved to {checkpoint_path}")
                    
                    # Save to W&B
                    if use_wandb:
                        models.train_unified.wandb.save(checkpoint_path)
                        
                        # Log model performance metrics
                        models.train_unified.wandb.run.summary["best_val_loss"] = best_val_loss
                        models.train_unified.wandb.run.summary["best_val_acc"] = val_direction_acc
                        models.train_unified.wandb.run.summary["best_epoch"] = epoch
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Print profiling summary
        profiler.print_summary()
        
        # Export chrome trace
        trace_path = os.path.join(profiler.profile_dir, "training_trace.json")
        profiler.export_chrome_trace(trace_path)
        
        # Save profiling data
        profiler.save_summary()
        
        # Save final model
        with profiler.profile_region("save_final_model"):
            final_checkpoint_path = f"{checkpoint_base}_final.pt"
            models.train_unified.torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, final_checkpoint_path)
        
        print(f"✅ Training complete! Final model saved to {final_checkpoint_path}")
        
        # Create performance visualizations for W&B
        if use_wandb:
            # Log final model
            models.train_unified.wandb.save(final_checkpoint_path)
            
            # Create performance plots for W&B dashboard
            try:
                import matplotlib.pyplot as plt
                
                # Loss curve
                plt.figure(figsize=(10, 5))
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title('Loss Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('/tmp/loss_curve.png')
                models.train_unified.wandb.log({"loss_curve": models.train_unified.wandb.Image('/tmp/loss_curve.png')})
                
                # Accuracy curve
                plt.figure(figsize=(10, 5))
                plt.plot(history['train_direction_acc'], label='Train Accuracy')
                plt.plot(history['val_direction_acc'], label='Validation Accuracy')
                plt.title('Direction Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('/tmp/accuracy_curve.png')
                models.train_unified.wandb.log({"accuracy_curve": models.train_unified.wandb.Image('/tmp/accuracy_curve.png')})
                
                # Learning rate curve
                plt.figure(figsize=(10, 5))
                plt.plot(history['learning_rate'], label='Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('/tmp/lr_curve.png')
                models.train_unified.wandb.log({"lr_curve": models.train_unified.wandb.Image('/tmp/lr_curve.png')})
                
                # Log profiling visualizations to W&B
                models.train_unified.wandb.log({
                    "profiling_timeline": models.train_unified.wandb.Image(os.path.join(profiler.profile_dir, "timeline.png")),
                    "memory_usage": models.train_unified.wandb.Image(os.path.join(profiler.profile_dir, "memory_usage.png")),
                })
            except Exception as e:
                print(f"Error creating plots: {e}")
        
        return model, history
    
    # Apply the monkey patch
    models.train_unified.train_optimized = patched_train_optimized
    
    return profiler


def integrate_with_run_training():
    """
    Add profiling support to run_training.py.
    """
    # Add profiler command line option to run_training.py
    # Modify the main function to use profiler
    pass  # Implementation would go here