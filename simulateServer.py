import os
import time
import random

SERVER_INFO_DIR = './server_info'
os.makedirs(SERVER_INFO_DIR, exist_ok=True)

def generate_server_info():
    for i in range(1, 7):
        file_path = os.path.join(SERVER_INFO_DIR, f'server_{i}_info.txt')
        
        # 模拟推理任务数量和总时间
        tasks_completed = random.randint(10, 1000)
        total_time = random.uniform(5.0, 500.0)
        timestamp = time.time()
        
        with open(file_path, 'w') as f:
            f.write(f"{tasks_completed},{total_time},{timestamp}")
        
        print(f"Generated info for Server {i}")

if __name__ == '__main__':
    while True:
        generate_server_info()
        time.sleep(5)  # 每5秒更新一次服务器信息