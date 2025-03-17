import os
import json
import time
import threading
import http.server
import socketserver
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Import the SimpleClusterManager from the cluster manager module
from serviceManager import SimpleClusterManager

# HTML template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Cluster Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .stats { margin-bottom: 20px; }
        .stats span { display: inline-block; margin-right: 15px; padding: 5px 10px; border-radius: 4px; }
        .total { background-color: #e9ecef; }
        .healthy { background-color: #d4edda; color: #155724; }
        .unhealthy { background-color: #f8d7da; color: #721c24; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        tr:hover { background-color: #f5f5f5; }
        th { background-color: #f8f9fa; border-top: 1px solid #ddd; }
        .node-healthy { background-color: #d4edda; }
        .node-unhealthy { background-color: #f8d7da; }
        .metrics { margin-top: 5px; font-size: 0.9em; color: #666; }
        .custom-metrics { margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px; }
        .custom-metric { margin: 2px 0; }
        .refresh-time { color: #666; font-size: 0.8em; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Simple Cluster Dashboard</h1>
        <p class="refresh-time">Last refreshed: <span id="refresh-time">{refresh_time}</span></p>
    </div>
    
    <div class="stats">
        <span class="total">Total: {total_count}</span>
        <span class="healthy">Healthy: {healthy_count}</span>
        <span class="unhealthy">Unhealthy: {unhealthy_count}</span>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Service ID</th>
                <th>Host:Port</th>
                <th>Status</th>
                <th>Uptime</th>
                <th>Last Update</th>
                <th>Metrics</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    
    <script>
        // Auto-refresh the page every 10 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 10000);
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard"""
    
    def __init__(self, *args, cluster_manager=None, **kwargs):
        self.cluster_manager = cluster_manager
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_dashboard()
        else:
            self.send_error(404)
            
    def send_dashboard(self):
        """Generate and send the dashboard HTML"""
        if not self.cluster_manager:
            self.send_error(500, "Cluster manager not initialized")
            return
            
        try:
            # Get metrics from cluster manager
            metrics = self.cluster_manager.get_all_metrics()
            counts = self.cluster_manager.get_service_count()
            
            # Generate table rows
            table_rows = ""
            for service_id, service_metrics in metrics.items():
                host = service_metrics.get('host', 'unknown')
                port = service_metrics.get('port', 0)
                status = service_metrics.get('status', 'unknown')
                uptime = service_metrics.get('uptime', 0)
                timestamp = service_metrics.get('timestamp', '')
                file_age = service_metrics.get('file_age', float('inf'))
                
                # Format uptime
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours}h {minutes}m {seconds}s"
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    timestamp_str = timestamp
                    
                # Determine row class based on health
                is_healthy = status == 'running' and file_age <= self.cluster_manager.node_timeout
                row_class = "node-healthy" if is_healthy else "node-unhealthy"
                status_display = "RUNNING" if is_healthy else "DOWN"
                
                # Format metrics
                cpu = service_metrics.get('cpu_usage', 0)
                memory = service_metrics.get('memory_usage', 0)
                connections = service_metrics.get('active_connections', 0)
                
                metrics_html = f"""
                <div class="metrics">
                    CPU: {cpu:.1f}% | Memory: {memory:.1f}% | Connections: {connections}
                </div>
                """
                
                # Format custom metrics
                custom_metrics = service_metrics.get('custom_metrics', {})
                if custom_metrics:
                    metrics_html += '<div class="custom-metrics">'
                    for key, value in custom_metrics.items():
                        metrics_html += f'<div class="custom-metric"><strong>{key}:</strong> {value}</div>'
                    metrics_html += '</div>'
                
                # Create row
                table_rows += f"""
                <tr class="{row_class}">
                    <td>{service_id}</td>
                    <td>{host}:{port}</td>
                    <td>{status_display}</td>
                    <td>{uptime_str}</td>
                    <td>{timestamp_str}</td>
                    <td>{metrics_html}</td>
                </tr>
                """
                
            # Generate complete HTML
            html = HTML_TEMPLATE.format(
                refresh_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_count=counts['total'],
                healthy_count=counts['healthy'],
                unhealthy_count=counts['unhealthy'],
                table_rows=table_rows
            )
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
            
        except Exception as e:
            self.send_error(500, str(e))


def run_dashboard(cluster_manager, host="localhost", port=8080):
    """Run the dashboard web server"""
    
    # Create custom handler with access to cluster manager
    handler = lambda *args, **kwargs: DashboardHandler(*args, cluster_manager=cluster_manager, **kwargs)
    
    # Create and start the server
    server = socketserver.TCPServer((host, port), handler)
    print(f"Dashboard running at http://{host}:{port}/")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# If run directly, start both cluster manager and dashboard
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Cluster Dashboard")
    parser.add_argument('--metrics-dir', default='/tmp/cluster_metrics', help="Directory for metrics files")
    parser.add_argument('--check-interval', type=int, default=5, help="Check interval in seconds")
    parser.add_argument('--host', default='localhost', help="Dashboard host")
    parser.add_argument('--port', type=int, default=8080, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Create and start the cluster manager
    manager = SimpleClusterManager(
        metrics_dir=args.metrics_dir,
        check_interval=args.check_interval
    )
    manager.start()
    
    # Run the dashboard
    try:
        run_dashboard(manager, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.stop()