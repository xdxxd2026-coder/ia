# monitor.py
import psutil
import GPUtil
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    def log_metrics(self):
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'gpu_load': self._get_gpu_usage(),
            'temperature': self._get_temperature()
        }
        self.metrics.append(metrics)
        return metrics
    
    def _get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            return [gpu.load for gpu in gpus]
        except:
            return []