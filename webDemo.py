import os
import io  # 使用 io 模块的 BytesIO
import json
import time
import random
import threading
import base64
import matplotlib.pyplot as plt
from http.server import HTTPServer, SimpleHTTPRequestHandler

# 确保状态目录存在
STATUS_DIR = 'server_status'
os.makedirs(STATUS_DIR, exist_ok=True)


def simulate_inference_server(server_name, stop_event):
    """
    模拟推理服务器的运行状态

    :param server_name: 服务器名称
    :param stop_event: 停止事件标志
    """
    status_file = os.path.join(STATUS_DIR, f'{server_name}_status.txt')
    total_tasks = 0
    total_time = 0

    while not stop_event.is_set():
        # 模拟推理任务
        task_count = random.randint(1, 10)
        task_time = random.uniform(0.1, 2.0)

        total_tasks += task_count
        total_time += task_time * task_count

        # 写入状态文件
        status = {
            'tasks_completed': total_tasks,
            'total_time': total_time,
            'timestamp': time.time()
        }

        with open(status_file, 'w') as f:
            json.dump(status, f)

        # 随机间隔，模拟实际工作
        time.sleep(random.uniform(1, 3))


class ServerMonitor:
    def __init__(self, status_dir):
        self.status_dir = status_dir
        self.server_history = {}

    def read_server_status(self):
        """读取所有服务器状态"""
        server_status = {}
        for filename in os.listdir(self.status_dir):
            if filename.endswith('_status.txt'):
                server_name = filename.split('_')[0]
                file_path = os.path.join(self.status_dir, filename)

                try:
                    with open(file_path, 'r') as f:
                        status = json.load(f)

                    server_status[server_name] = {
                        'tasks_completed': status.get('tasks_completed', 0),
                        'total_time': status.get('total_time', 0),
                        'avg_task_time': status.get('total_time', 0) / status.get('tasks_completed', 1)
                        if status.get('tasks_completed', 0) > 0 else 0,
                        'timestamp': status.get('timestamp', time.time())
                    }

                    # 记录历史数据
                    if server_name not in self.server_history:
                        self.server_history[server_name] = []

                    self.server_history[server_name].append({
                        'tasks_completed': status.get('tasks_completed', 0),
                        'avg_task_time': status.get('total_time', 0) / status.get('tasks_completed', 1)
                        if status.get('tasks_completed', 0) > 0 else 0,
                        'timestamp': status.get('timestamp', time.time())
                    })

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {filename}: {e}")

        return server_status

    def generate_plots(self):
        """生成服务器监控图表"""
        plots = {}
        plt.close('all')

        for server_name, history in self.server_history.items():
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle(f'{server_name} 服务器监控')

            # 任务完成数
            tasks_completed = [entry['tasks_completed'] for entry in history]
            ax1.plot(tasks_completed, marker='o')
            ax1.set_title('完成任务数')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('任务数')

            # 平均任务时间
            avg_task_times = [entry['avg_task_time'] for entry in history]
            ax2.plot(avg_task_times, marker='o', color='red')
            ax2.set_title('平均任务耗时')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('平均时间(s)')

            plt.tight_layout()

            # 将图形转换为 Base64
            buffer = io.BytesIO()  # 使用 io.BytesIO
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plots[server_name] = image_base64

            plt.close(fig)

        return plots


# 后续代码保持不变（InferenceMonitorHandler 和 main 函数）
# ...（此处省略其他部分，与之前的代码相同）
class InferenceMonitorHandler(SimpleHTTPRequestHandler):
    monitor = ServerMonitor(STATUS_DIR)

    def do_GET(self):
        """处理 GET 请求"""
        if self.path == '/':
            self.send_dashboard_page()
        elif self.path == '/server_status':
            self.send_server_status()
        else:
            super().do_GET()

    def do_POST(self):
        """处理 POST 请求"""
        if self.path == '/stop_services':
            self.stop_all_services()
        else:
            self.send_error(404)

    def send_dashboard_page(self):
        """发送仪表板 HTML 页面"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

        html_content = self.generate_dashboard_html()
        self.wfile.write(html_content.encode('utf-8'))

    def generate_dashboard_html(self):
        """生成仪表板 HTML"""
        current_status = self.monitor.read_server_status()
        plots = self.monitor.generate_plots()

        # 构建服务器卡片的 HTML
        server_cards_html = ""
        for server_name, status in current_status.items():
            server_cards_html += f'''
            <div class="server-card">
                <h3>{server_name} 服务器</h3>
                <p>完成任务数: {status['tasks_completed']}</p>
                <p>总耗时: {status['total_time']:.2f}s</p>
                <p>平均任务耗时: {status['avg_task_time']:.2f}s</p>
                <img src="data:image/png;base64,{plots.get(server_name, '')}" alt="{server_name} 监控图">
            </div>
            '''

        return f'''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>推理服务器监控</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .dashboard {{ 
                    display: grid; 
                    grid-template-columns: repeat(3, 1fr); 
                    gap: 20px; 
                    padding: 20px;
                }}
                .server-card {{ 
                    border: 1px solid #ddd; 
                    padding: 10px; 
                    text-align: center; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .server-card img {{ 
                    max-width: 100%; 
                    height: auto; 
                }}
                .stop-button {{ 
                    display: block; 
                    width: 200px; 
                    margin: 20px auto; 
                    padding: 10px; 
                    background-color: #f44336; 
                    color: white; 
                    border: none; 
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <h1 style="text-align: center;">推理服务器监控</h1>
            <div class="dashboard">
                {server_cards_html}
            </div>
            <button class="stop-button" onclick="stopServices()">停止所有推理服务</button>

            <script>
                function stopServices() {{
                    fetch('/stop_services', {{ method: 'POST' }})
                        .then(response => response.json())
                        .then(data => {{ alert('已发送停止服务信号'); }});
                }}
            </script>
        </body>
        </html>
        '''

    def send_server_status(self):
        """发送服务器状态 JSON"""
        current_status = self.monitor.read_server_status()
        plots = self.monitor.generate_plots()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        response_data = json.dumps({
            'current_status': current_status,
            'plots': plots
        })
        self.wfile.write(response_data.encode('utf-8'))

    def stop_all_services(self):
        """处理停止服务请求"""
        with open('stop_signal.txt', 'w') as f:
            json.dump({'stop_all_services': True}, f)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        response_data = json.dumps({'status': 'stop signal sent'})
        self.wfile.write(response_data.encode('utf-8'))
def main():
    # matplotlib 后端设置
    plt.switch_backend('Agg')

    # 创建停止事件
    stop_event = threading.Event()

    # 模拟多个推理服务器
    server_names = ['server1', 'server2', 'server3', 'server4', 'server5', 'server6']
    server_threads = []

    for server_name in server_names:
        thread = threading.Thread(
            target=simulate_inference_server,
            args=(server_name, stop_event)
        )
        thread.start()
        server_threads.append(thread)

    try:
        # 启动 Web 服务器
        server_address = ('', 8000)
        httpd = HTTPServer(server_address, InferenceMonitorHandler)
        print('服务器运行在 http://localhost:8000')
        httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n正在停止服务...")
        stop_event.set()  # 设置停止事件

        # 等待所有服务器线程结束
        for thread in server_threads:
            thread.join()


if __name__ == '__main__':
    main()