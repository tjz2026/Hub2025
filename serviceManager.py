import os
import json
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_cluster_manager')

class SimpleServiceNode:
    """
    Client that runs on each service host to report metrics by writing to a file.
    """
    
    def __init__(self, 
                 service_id: str,
                 service_port: int,
                 metrics_dir: str = '/tmp/cluster_metrics',
                 report_interval: int = 10):
        """
        Initialize the Service Node reporter.
        
        Args:
            service_id: Unique identifier for this service
            service_port: Port the service is running on
            metrics_dir: Directory to write metrics files
            report_interval: Interval in seconds for reporting metrics
        """
        self.service_id = service_id
        self.host = os.uname()[1]  # Get hostname
        self.service_port = service_port
        self.metrics_dir = Path(metrics_dir)
        self.metrics_file = self.metrics_dir / f"{service_id}.json"
        self.report_interval = report_interval
        
        self.start_time = time.time()
        self.custom_metrics = {}
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.running = False
        self.reporter_thread = None

    def start(self) -> None:
        """Start the metrics reporter"""
        if self.running:
            return
            
        self.running = True
        self.reporter_thread = threading.Thread(target=self._reporter_thread)
        self.reporter_thread.daemon = True
        self.reporter_thread.start()
        
        logger.info(f"ServiceNode reporter started for {self.service_id}")

    def stop(self) -> None:
        """Stop the metrics reporter"""
        self.running = False
        if self.reporter_thread:
            self.reporter_thread.join(timeout=5.0)
        
        # Remove metrics file when stopping
        try:
            if self.metrics_file.exists():
                self.metrics_file.unlink()
        except Exception as e:
            logger.error(f"Error removing metrics file: {e}")
            
        logger.info(f"ServiceNode reporter stopped for {self.service_id}")

    def _reporter_thread(self) -> None:
        """Thread that periodically writes metrics to file"""
        while self.running:
            try:
                self._write_metrics()
                time.sleep(self.report_interval)
            except Exception as e:
                logger.error(f"Error writing metrics: {e}")
                time.sleep(5)  # Sleep before retry on error

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the local service"""
        metrics = {
            'service_id': self.service_id,
            'host': self.host,
            'port': self.service_port,
            'status': 'running',
            'uptime': int(time.time() - self.start_time),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Basic system metrics - using psutil if available, otherwise estimate
            try:
                import psutil
                process = psutil.Process(os.getpid())
                metrics['cpu_usage'] = process.cpu_percent()
                metrics['memory_usage'] = process.memory_percent()
                metrics['active_connections'] = len(process.connections())
            except ImportError:
                # Fallback without psutil
                metrics['cpu_usage'] = 0.0
                metrics['memory_usage'] = 0.0
                metrics['active_connections'] = 0
                
            # Add any custom metrics
            metrics['custom_metrics'] = self.custom_metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            metrics['status'] = 'error'
            
        return metrics

    def _write_metrics(self) -> None:
        """Write metrics to file"""
        metrics = self._collect_metrics()
        
        # Write to a temporary file first, then move it to avoid partial reads
        temp_file = self.metrics_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(metrics, f)
            
            # Atomic move
            shutil.move(temp_file, self.metrics_file)
            
        except Exception as e:
            logger.error(f"Error writing metrics file: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def update_custom_metric(self, key: str, value: Any) -> None:
        """Update a custom metric"""
        self.custom_metrics[key] = value

    def remove_custom_metric(self, key: str) -> None:
        """Remove a custom metric"""
        if key in self.custom_metrics:
            del self.custom_metrics[key]


class SimpleClusterManager:
    """
    Simple cluster manager that monitors services by reading metrics files.
    """
    
    def __init__(self, 
                 metrics_dir: str = '/tmp/cluster_metrics',
                 check_interval: int = 10,
                 node_timeout: int = 30):
        """
        Initialize the Simple Cluster Manager.
        
        Args:
            metrics_dir: Directory where metrics files are stored
            check_interval: Interval in seconds for checking metrics
            node_timeout: Time in seconds after which a node is considered down
        """
        self.metrics_dir = Path(metrics_dir)
        self.check_interval = check_interval
        self.node_timeout = node_timeout
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.metrics_cache = {}
        self.lock = threading.RLock()
        
        self.running = False
        self.collector_thread = None

    def start(self) -> None:
        """Start the cluster manager"""
        if self.running:
            return
            
        self.running = True
        self.collector_thread = threading.Thread(target=self._collector_thread)
        self.collector_thread.daemon = True
        self.collector_thread.start()
        
        logger.info(f"Simple cluster manager started (monitoring {self.metrics_dir})")

    def stop(self) -> None:
        """Stop the cluster manager"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
            
        logger.info("Simple cluster manager stopped")

    def _collector_thread(self) -> None:
        """Thread that periodically reads metrics files"""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                time.sleep(5)  # Sleep before retry on error

    def _collect_metrics(self) -> None:
        """Read all metrics files from the metrics directory"""
        current_time = datetime.now()
        
        # Get list of metrics files
        metrics_files = list(self.metrics_dir.glob('*.json'))
        
        # Track existing services
        existing_services = set()
        
        for file_path in metrics_files:
            service_id = file_path.stem
            existing_services.add(service_id)
            
            try:
                # Check file modification time first
                file_mtime = file_path.stat().st_mtime
                file_age = current_time.timestamp() - file_mtime
                
                # If file hasn't been updated recently, mark service as down
                if file_age > self.node_timeout:
                    with self.lock:
                        if service_id in self.metrics_cache:
                            self.metrics_cache[service_id]['status'] = 'down'
                            self.metrics_cache[service_id]['file_age'] = file_age
                    continue
                
                # Read metrics from file
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                
                # Add file age to metrics
                metrics['file_age'] = file_age
                
                # Update cache
                with self.lock:
                    self.metrics_cache[service_id] = metrics
                    
            except Exception as e:
                logger.error(f"Error reading metrics file for {service_id}: {e}")
                
        # Mark missing services as down
        with self.lock:
            for service_id in list(self.metrics_cache.keys()):
                if service_id not in existing_services:
                    # Service file no longer exists
                    self.metrics_cache[service_id]['status'] = 'down'
                    self.metrics_cache[service_id]['file_age'] = float('inf')

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all services"""
        with self.lock:
            # Return a copy to avoid concurrent modification issues
            return {k: v.copy() for k, v in self.metrics_cache.items()}

    def get_service_metrics(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific service"""
        with self.lock:
            if service_id in self.metrics_cache:
                return self.metrics_cache[service_id].copy()
            return None

    def get_unhealthy_services(self) -> List[Dict[str, Any]]:
        """Get a list of unhealthy services"""
        unhealthy = []
        with self.lock:
            for service_id, metrics in self.metrics_cache.items():
                # Consider a service unhealthy if it's down or file is too old
                if metrics.get('status') != 'running' or metrics.get('file_age', 0) > self.node_timeout:
                    unhealthy.append(metrics.copy())
        return unhealthy

    def get_service_count(self) -> Dict[str, int]:
        """Get count of total, healthy, and unhealthy services"""
        total = 0
        healthy = 0
        unhealthy = 0
        
        with self.lock:
            for service_id, metrics in self.metrics_cache.items():
                total += 1
                # Consider a service healthy if it's running and file is fresh
                if metrics.get('status') == 'running' and metrics.get('file_age', float('inf')) <= self.node_timeout:
                    healthy += 1
                else:
                    unhealthy += 1
                    
        return {
            'total': total,
            'healthy': healthy,
            'unhealthy': unhealthy
        }


# Simple command-line interface for the cluster manager
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Simple Cluster Manager")
    parser.add_argument('mode', choices=['manager', 'node'], help="Run as manager or node")
    parser.add_argument('--metrics-dir', default='/tmp/cluster_metrics', help="Directory for metrics files")
    parser.add_argument('--interval', type=int, default=10, help="Check/report interval in seconds")
    parser.add_argument('--service-id', help="Service ID (required for node mode)")
    parser.add_argument('--service-port', type=int, help="Service port (required for node mode)")
    
    args = parser.parse_args()
    
    # Ensure metrics directory exists
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    if args.mode == 'manager':
        # Run as cluster manager
        manager = SimpleClusterManager(
            metrics_dir=args.metrics_dir,
            check_interval=args.interval
        )
        
        try:
            manager.start()
            print(f"Cluster manager monitoring {args.metrics_dir}")
            
            # Simple CLI for monitoring
            while True:
                time.sleep(args.interval)
                counts = manager.get_service_count()
                print(f"\n--- Services: {counts['total']} total, {counts['healthy']} healthy, {counts['unhealthy']} unhealthy ---")
                
                metrics = manager.get_all_metrics()
                for service_id, service_metrics in metrics.items():
                    status = "HEALTHY" if service_metrics.get('status') == 'running' and service_metrics.get('file_age', float('inf')) <= manager.node_timeout else "UNHEALTHY"
                    print(f"{service_id} ({service_metrics.get('host', 'unknown')}:{service_metrics.get('port', 0)}) - {status}")
                    
        except KeyboardInterrupt:
            print("\nShutting down cluster manager...")
            manager.stop()
            
    elif args.mode == 'node':
        if not args.service_id or not args.service_port:
            parser.error("--service-id and --service-port are required for node mode")
        
        # Run as service node
        node = SimpleServiceNode(
            service_id=args.service_id,
            service_port=args.service_port,
            metrics_dir=args.metrics_dir,
            report_interval=args.interval
        )
        
        try:
            node.start()
            print(f"Service node reporter running for {args.service_id}")
            
            # Example of updating custom metrics
            count = 0
            while True:
                count += 1
                node.update_custom_metric("example_counter", count)
                node.update_custom_metric("last_action_time", datetime.now().isoformat())
                print(f"Updated metrics for {args.service_id} (count: {count})")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nShutting down service node reporter...")
            node.stop()