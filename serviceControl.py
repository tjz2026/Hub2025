import os
import socket
import time
import subprocess
import sys
import signal
import fcntl
import errno
from pathlib import Path

class ServiceManager:
    """
    Manages a service that can be started by clients in a competing scenario.
    Only one client will successfully start the service while others will wait.
    """
    
    def __init__(self, 
                 service_host='localhost', 
                 service_port=12345, 
                 lock_file_path='/tmp/service_lock',
                 service_script_path='service.py',
                 connection_timeout=0.5,
                 startup_timeout=10,
                 startup_check_interval=0.2):
        """
        Initialize the ServiceManager.
        
        Args:
            service_host: Host where the service runs
            service_port: Port the service listens on
            lock_file_path: Path to the lock file for coordinating service startup
            service_script_path: Path to the Python script that starts the service
            connection_timeout: Timeout for connection attempts in seconds
            startup_timeout: Maximum time to wait for service to start in seconds
            startup_check_interval: Interval between connection attempts during startup
        """
        self.service_host = service_host
        self.service_port = service_port
        self.lock_file_path = Path(lock_file_path)
        self.service_script_path = service_script_path
        self.connection_timeout = connection_timeout
        self.startup_timeout = startup_timeout
        self.startup_check_interval = startup_check_interval
        
    def is_service_running(self):
        """Check if the service is already running by attempting a connection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.connection_timeout)
                s.connect((self.service_host, self.service_port))
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
            
    def acquire_lock(self):
        """
        Try to acquire the lock file. Return True if successful, False otherwise.
        """
        try:
            # Create the lock file if it doesn't exist
            self.lock_fd = os.open(self.lock_file_path, os.O_CREAT | os.O_RDWR)
            
            # Try to acquire an exclusive lock
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write the current PID to the lock file
            os.truncate(self.lock_fd, 0)
            os.write(self.lock_fd, str(os.getpid()).encode())
            
            return True
            
        except IOError as e:
            # Lock couldn't be acquired (another process has it)
            if e.errno == errno.EAGAIN:
                if hasattr(self, 'lock_fd'):
                    os.close(self.lock_fd)
                return False
            raise
            
    def release_lock(self):
        """Release the lock file if we have it."""
        if hasattr(self, 'lock_fd'):
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            
    def start_service(self):
        """
        Start the service as a subprocess.
        """
        try:
            # Start the service in a new process group
            process = subprocess.Popen(
                [sys.executable, self.service_script_path],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Write the service process ID to the lock file
            os.truncate(self.lock_fd, 0)
            os.write(self.lock_fd, f"{os.getpid()}:{process.pid}".encode())
            
            return process.pid
        except Exception as e:
            print(f"Failed to start service: {e}")
            self.release_lock()
            raise
            
    def wait_for_service(self):
        """
        Wait for the service to become available, with timeout.
        Return True if service is available, False otherwise.
        """
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            if self.is_service_running():
                return True
            time.sleep(self.startup_check_interval)
            
        return False
        
    def connect_or_start_service(self):
        """
        Main method to connect to the service or start it if not running.
        
        Returns:
            tuple: (socket_connection, started_by_this_client)
        """
        # Fast path: Check if service is already running
        if self.is_service_running():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.service_host, self.service_port))
            return sock, False
            
        # Slow path: Service not running, try to acquire lock and start it
        if self.acquire_lock():
            # Double-check that the service isn't running
            # (another client might have started it between our check and lock)
            if self.is_service_running():
                self.release_lock()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.service_host, self.service_port))
                return sock, False
                
            # We have the lock and the service isn't running - start it
            try:
                service_pid = self.start_service()
                print(f"Started service with PID {service_pid}")
                
                # Wait for the service to become available
                if self.wait_for_service():
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((self.service_host, self.service_port))
                    return sock, True
                else:
                    raise TimeoutError("Service failed to start within timeout period")
                    
            finally:
                # Keep the lock file around but release our lock
                self.release_lock()
        else:
            # We couldn't acquire the lock, someone else is starting the service
            print("Another client is starting the service, waiting...")
            if self.wait_for_service():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.service_host, self.service_port))
                return sock, False
            else:
                raise TimeoutError("Timed out waiting for another client to start the service")


# Example usage:
if __name__ == "__main__":
    # Example client code
    try:
        manager = ServiceManager(
            service_host='localhost',
            service_port=12345,
            lock_file_path='/tmp/my_service_lock',
            service_script_path='./my_service.py'
        )
        
        connection, started = manager.connect_or_start_service()
        
        if started:
            print("This client started the service")
        else:
            print("Connected to an existing service")
            
        # Use the connection...
        connection.send(b"Hello, service!")
        response = connection.recv(1024)
        print(f"Received: {response.decode()}")
        
        # Close the connection when done
        connection.close()
        
    except Exception as e:
        print(f"Error: {e}")