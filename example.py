# my_service.py - Example service implementation
import socket
import signal
import sys
import time
import os

def shutdown_handler(signum, frame):
    """Handler for graceful shutdown on signals"""
    print("Service shutting down...")
    if 'server_socket' in globals():
        server_socket.close()
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# Create server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Allow socket to be reused immediately after closing
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind to localhost:12345
server_socket.bind(('localhost', 12345))
server_socket.listen(5)

print(f"Service running on port 12345 with PID {os.getpid()}")

try:
    while True:
        # Accept client connections
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        
        # Handle client request
        data = client_socket.recv(1024)
        if data:
            print(f"Received: {data.decode()}")
            # Echo back with acknowledgment
            response = f"Service received: {data.decode()}"
            client_socket.send(response.encode())
            
        # Close client connection
        client_socket.close()
except KeyboardInterrupt:
    print("Service interrupted")
finally:
    server_socket.close()
    print("Service stopped")