import http.server
import socketserver
import json
import os
import datetime
import threading
import time
import urllib.parse
import html

# Configuration
JSON_FILE_PATH = "data.json"  # Path to your JSON file
UPDATE_INTERVAL = 30  # How often to check for updates (in seconds)
PORT = 8000  # Web server port

# Global variables to store data
data_cache = {}
last_modified_time = 0
last_update_time = None

def load_json_data():
    """Load data from JSON file and update the cache"""
    global data_cache, last_modified_time, last_update_time
    
    try:
        # Check if file exists
        if not os.path.exists(JSON_FILE_PATH):
            data_cache = {"error": "JSON file not found"}
            return
            
        # Check if file has been modified
        current_mtime = os.path.getmtime(JSON_FILE_PATH)
        if current_mtime > last_modified_time:
            with open(JSON_FILE_PATH, 'r') as file:
                data_cache = json.load(file)
            last_modified_time = current_mtime
            last_update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        data_cache = {"error": f"Failed to load JSON: {str(e)}"}

def update_data_periodically():
    """Background thread to update data periodically"""
    while True:
        load_json_data()
        time.sleep(UPDATE_INTERVAL)

def format_value(value):
    """Format a value for HTML display"""
    if isinstance(value, (dict, list)):
        return f"<pre>{html.escape(json.dumps(value, indent=2))}</pre>"
    else:
        return html.escape(str(value))

def generate_html():
    """Generate HTML for the dashboard"""
    error = data_cache.get("error")
    
    # Create the HTML content as a single string without format() method
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Simple JSON Dashboard</title>
    <meta http-equiv="refresh" content="30"> <!-- Auto-refresh page every 30 seconds -->
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .update-info {{ margin-bottom: 20px; color: #666; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>JSON Data Dashboard</h1>
    
    <div class="update-info">
        Last updated: {last_update_time or "Never"}
        <button onclick="location.reload()">Refresh Now</button>
    </div>
    """
    
    # Add error message if there's an error
    if error:
        html_content += f'<p class="error">{html.escape(error)}</p>'
    else:
        # Display data as a table
        if isinstance(data_cache, dict):
            html_content += """
            <table>
                <tr><th>Key</th><th>Value</th></tr>
            """
            for key, value in data_cache.items():
                html_content += f"""
                <tr>
                    <td>{html.escape(str(key))}</td>
                    <td>{format_value(value)}</td>
                </tr>
                """
            html_content += "</table>"
        elif isinstance(data_cache, list):
            html_content += """
            <table>
                <tr><th>#</th><th>Value</th></tr>
            """
            for i, item in enumerate(data_cache):
                html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{format_value(item)}</td>
                </tr>
                """
            html_content += "</table>"
        else:
            html_content += f"<p>{html.escape(str(data_cache))}</p>"
    
    # Close the HTML tags
    html_content += """
</body>
</html>
    """
    
    return html_content

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Only respond to the root path
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(generate_html().encode('utf-8'))
        else:
            # Handle 404 for other paths
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'404 Not Found')

def start_server():
    """Start the HTTP server"""
    handler = DashboardHandler
    
    # Avoid "Address already in use" error
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Load data once before starting
    load_json_data()
    
    # Start the background update thread
    update_thread = threading.Thread(target=update_data_periodically, daemon=True)
    update_thread.start()
    
    # Start the web server
    start_server()