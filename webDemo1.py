# @Author: Jiuzhou Tang
# @Time : 2025/3/28 7:38
import os
import json
import time
import random
import threading
import http.server
import socketserver
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cgi

# 服务器信息模拟和读取配置
SERVER_INFO_DIR = './server_info'
SERVER_STATUS_FILE = './server_status.txt'
NUM_SERVERS = 6
MAX_HISTORY_POINTS = 50
WEBSOCKET_PORT = 8765


class InferenceMonitor:
    def __init__(self):
        self.server_data = {f'Server {i + 1}': {
            'tasks_completed': [],
            'total_time': [],
            'timestamps': []
        } for i in range(NUM_SERVERS)}

    def read_server_info(self):
        for i in range(1, NUM_SERVERS + 1):
            try:
                file_path = os.path.join(SERVER_INFO_DIR, f'server_{i}_info.txt')
                with open(file_path, 'r') as f:
                    data = f.read().strip().split(',')
                    tasks = int(data[0])
                    total_time = float(data[1])
                    timestamp = float(data[2])

                server_key = f'Server {i}'
                self.server_data[server_key]['tasks_completed'].append(tasks)
                self.server_data[server_key]['total_time'].append(total_time)
                self.server_data[server_key]['timestamps'].append(timestamp)

                # 保持历史记录在最大点数范围内
                if len(self.server_data[server_key]['tasks_completed']) > MAX_HISTORY_POINTS:
                    self.server_data[server_key]['tasks_completed'] = self.server_data[server_key]['tasks_completed'][
                                                                      -MAX_HISTORY_POINTS:]
                    self.server_data[server_key]['total_time'] = self.server_data[server_key]['total_time'][
                                                                 -MAX_HISTORY_POINTS:]
                    self.server_data[server_key]['timestamps'] = self.server_data[server_key]['timestamps'][
                                                                 -MAX_HISTORY_POINTS:]

            except Exception as e:
                print(f"Error reading server {i} info: {e}")

        return self.server_data

    def generate_plots(self):
        plt.close('all')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ML Inference Servers Performance', fontsize=16)

        for i, (server, data) in enumerate(self.server_data.items()):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            tasks = data['tasks_completed']
            times = data['total_time']

            # 主线图：任务完成数量
            ax.plot(tasks, label='Tasks Completed', color='blue')
            ax.set_title(server)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Tasks')

            # 如果有数据，添加平均时间子图
            if times:
                avg_time = np.mean(times)
                inset_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                inset_ax.bar(['Avg Time'], [avg_time], color='red')
                inset_ax.set_title('Avg Task Time')
                inset_ax.set_ylabel('Seconds')

        plt.tight_layout()

        # 将图转换为 base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64


# 自定义 HTTP 请求处理器
class MLInferenceHandler(http.server.SimpleHTTPRequestHandler):
    monitor = InferenceMonitor()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        elif self.path == '/plot':
            # 生成并返回性能图
            self.monitor.read_server_info()
            plot_image = self.monitor.generate_plots()

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(plot_image.encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/stop_servers':
            try:
                with open(SERVER_STATUS_FILE, 'w') as f:
                    f.write('STOP_ALL_SERVERS')

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'sent'}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())


# HTML 内容
HTML_CONTENT = '''
<!DOCTYPE html>
<html>
<head>
    <title>ML Inference Server Monitor</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            background-color: #f4f4f4; 
        }
        #dashboard-plot { 
            max-width: 90%; 
            margin: 20px auto; 
        }
        #stop-servers-btn {
            background-color: #ff4444;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>ML Inference Servers Dashboard</h1>
    <button id="stop-servers-btn">Stop All Servers</button>
    <div id="dashboard-plot"></div>

    <script>
        const dashboardPlot = document.getElementById('dashboard-plot');
        const stopServersBtn = document.getElementById('stop-servers-btn');

        function updateDashboard() {
            fetch('/plot')
                .then(response => response.text())
                .then(plotData => {
                    dashboardPlot.innerHTML = `<img src="data:image/png;base64,${plotData}" alt="Server Performance">`;
                });
        }

        // 初始化和定期更新
        updateDashboard();
        setInterval(updateDashboard, 5000);

        stopServersBtn.addEventListener('click', () => {
            fetch('/stop_servers', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'sent') {
                        alert('停止服务器指令已发送');
                    } else {
                        alert('发送指令时发生错误: ' + data.message);
                    }
                });
        });
    </script>
</body>
</html>
'''


def run_server(port=8001):
    with socketserver.TCPServer(("", port), MLInferenceHandler) as httpd:
        print(f"服务器运行在 http://localhost:{port}")
        httpd.serve_forever()


if __name__ == '__main__':
    # 确保服务器信息目录存在
    os.makedirs(SERVER_INFO_DIR, exist_ok=True)

    # 运行 Web 服务器
    run_server()