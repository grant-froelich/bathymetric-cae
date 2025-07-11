# tests/utils/performance_monitor.py

import time
import psutil
import threading
from contextlib import contextmanager


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.max_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    @contextmanager
    def monitor(self, sample_interval=0.1):
        """Context manager for performance monitoring."""
        self.start_monitoring(sample_interval)
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, sample_interval=0.1):
        """Start performance monitoring."""
        self.reset()
        self.start_time = time.time()
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor,
            args=(sample_interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _memory_monitor(self, interval):
        """Monitor memory usage in separate thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                self.max_memory = max(self.max_memory, memory_mb)
                time.sleep(interval)
            except Exception:
                break
    
    def get_results(self):
        """Get monitoring results."""
        execution_time = None
        if self.start_time and self.end_time:
            execution_time = self.end_time - self.start_time
        
        return {
            'execution_time': execution_time,
            'max_memory_mb': self.max_memory,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'memory_samples': len(self.memory_samples)
        }


class GPUMonitor:
    """Monitor GPU performance metrics if available."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.reset()
    
    def _check_gpu_availability(self):
        """Check if GPU monitoring is available."""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            return False
    
    def reset(self):
        self.gpu_memory_samples = []
        self.max_gpu_memory = 0
    
    @contextmanager
    def monitor_gpu(self, sample_interval=0.5):
        """Monitor GPU usage during execution."""
        if not self.gpu_available:
            yield None
            return
        
        self.reset()
        monitoring = True
        
        def gpu_monitor():
            import tensorflow as tf
            while monitoring:
                try:
                    if tf.config.list_physical_devices('GPU'):
                        memory_info = tf.config.experimental.get_memory_info('GPU:0')
                        current_mb = memory_info['current'] / 1024 / 1024
                        self.gpu_memory_samples.append(current_mb)
                        self.max_gpu_memory = max(self.max_gpu_memory, current_mb)
                    time.sleep(sample_interval)
                except Exception:
                    break
        
        monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            yield self
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)
    
    def get_gpu_results(self):
        """Get GPU monitoring results."""
        if not self.gpu_available:
            return None
        
        return {
            'max_gpu_memory_mb': self.max_gpu_memory,
            'avg_gpu_memory_mb': sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            'gpu_samples': len(self.gpu_memory_samples)
        }


class BenchmarkTimer:
    """Simple benchmark timer for test performance measurement."""
    
    def __init__(self):
        self.times = {}
    
    @contextmanager
    def time_operation(self, operation_name):
        """Time a specific operation."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.times[operation_name] = end_time - start_time
    
    def get_times(self):
        """Get all recorded times."""
        return self.times.copy()
    
    def get_time(self, operation_name):
        """Get time for specific operation."""
        return self.times.get(operation_name)
    
    def clear(self):
        """Clear all recorded times."""
        self.times.clear()
    
    def summary(self):
        """Get summary of all timings."""
        if not self.times:
            return "No timings recorded"
        
        summary_lines = ["Performance Summary:"]
        for operation, duration in sorted(self.times.items()):
            summary_lines.append(f"  {operation}: {duration:.4f}s")
        
        total_time = sum(self.times.values())
        summary_lines.append(f"  Total: {total_time:.4f}s")
        
        return "\n".join(summary_lines)