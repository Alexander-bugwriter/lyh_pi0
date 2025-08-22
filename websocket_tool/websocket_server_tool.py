"""
极简WebSocket服务器 - 纯数据转发
接收: {"observation/image": np.array, "observation/state": np.array, "prompt": str, "reset": bool}
转发: 原样转发给推理端
"""

import asyncio
import websockets
import threading
import queue
import time
import sys
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import msgpack_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleForwardServer:
    """极简数据转发服务器"""
    
    def __init__(self):
        self.observation_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self._client_connected = False
    
    async def handle_client(self, websocket, path=None):
        """处理客户端连接"""
        logger.info(f"客户端连接: {websocket.remote_address}")
        self._client_connected = True
        
        try:
            # 必须先发送服务器元数据（OpenPI协议要求）
            server_metadata = {
                "server_type": "simple_forwarder",
                "protocol_version": "1.0",
                "supports_reset_flag": True
            }
            await websocket.send(msgpack_numpy.packb(server_metadata))
            logger.info("已发送服务器元数据")
            # 消息处理循环
            async for message in websocket:
                try:
                    # 解析并直接转发数据
                    data = msgpack_numpy.unpackb(message)
                    reset_flag = data.get("reset", False)
                    
                    logger.info(f"转发数据 - reset: {reset_flag}, prompt: {data.get('prompt', 'N/A')}")
                    
                    # 直接转发原始数据
                    self.observation_queue.put(data)
                    
                    # 等待策略响应
                    response = self.action_queue.get(timeout=30)
                    
                    # 发送响应
                    if hasattr(response, 'detach'):  # torch tensor
                        actions_np = response.detach().cpu().numpy()
                        if len(actions_np.shape) == 1:
                            actions_np = actions_np.reshape(1, -1)
                        formatted_response = {"actions": actions_np.tolist()}
                    else:
                        formatted_response = response
                    
                    await websocket.send(msgpack_numpy.packb(formatted_response))
                    logger.info("已发送动作响应")
                    
                except queue.Empty:
                    logger.error("策略响应超时")
                    await websocket.send(msgpack_numpy.packb({"error": "timeout"}))
                except Exception as e:
                    logger.error(f"处理消息错误: {e}")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("客户端断开连接")
        finally:
            self._client_connected = False
    
    async def run_server(self, host, port):
        """运行服务器"""
        logger.info(f"启动数据转发服务器: {host}:{port}")
        
        async with websockets.serve(
            self.handle_client, 
            host, 
            port, 
            compression=None,
            max_size=None,
            ping_interval=20,
            ping_timeout=10
        ):
            await asyncio.Future()
    
    def start_server(self, host="0.0.0.0", port=8000):
        """在后台线程启动服务器"""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_server(host, port))
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(2)
        logger.info("服务器已在后台启动")
        return thread
    
    def wait_for_observation(self):
        """等待观测数据"""
        return self.observation_queue.get()
    
    def send_action(self, actions):
        """发送动作"""
        self.action_queue.put(actions)
    
    def wait_for_client(self):
        """等待客户端连接"""
        while not self._client_connected:
            time.sleep(0.1)


# 全局接口
_server = None

def start_websocket_server(host="0.0.0.0", port=8000, device=None):
    """启动服务器（device 参数保留兼容性但不使用）"""
    global _server
    _server = SimpleForwardServer()
    _server.start_server(host, port)
    return _server

def wait_for_observation():
    return _server.wait_for_observation()

def wait_for_client_connection():
    _server.wait_for_client()

def send_action_response(actions):
    _server.send_action(actions)


if __name__ == "__main__":
    # 测试服务器
    server = start_websocket_server()
    wait_for_client_connection()
    
    logger.info("数据转发服务器就绪...")
    
    while True:
        # 接收原始数据
        raw_data = wait_for_observation()
        
        needs_reset = raw_data.get("reset", False)
        logger.info(f"收到数据 - reset: {needs_reset}, prompt: {raw_data.get('prompt', 'N/A')}")
        
        # 模拟策略输出
        import torch
        dummy_actions = torch.randn(5, 7) * 0.1
        send_action_response(dummy_actions)
