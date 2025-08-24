# Here’s where you define your tools (functions the AI can use)
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent
import paramiko
import logging
import asyncio
from pathlib import Path
import uuid
from typing import Literal, Optional, Dict, Any, List, Union

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
mcp = FastMCP("SSHServer", host="0.0.0.0", port=8000)


# SSH连接管理器
class SSHConnectionManager:
    
    def __init__(self):
        self.connections = {}
    
    def load_ssh_config(host_alias: str) -> dict:
        """解析 ~/.ssh/config 获取指定主机的配置"""
        ssh_config = paramiko.SSHConfig()
        config_path = Path.home() / ".ssh/config"
        
        if config_path.exists():
            with open(config_path) as f:
                ssh_config.parse(f)
            return ssh_config.lookup(host_alias)  # 返回字典格式的配置
        else:
            raise FileNotFoundError("SSH config file not found")
    
    async def create_connection(self, hostname: str, port: int,
                               username: str, password: str = None, key_path: str = None):
        """创建持久SSH连接并存储到会话状态"""

        session_id = str(uuid.uuid4())  # 生成唯一会话ID

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            if key_path:
                private_key = paramiko.RSAKey.from_private_key_file(key_path)
                ssh.connect(hostname, port, username, pkey=private_key)
            else:
                ssh.connect(hostname, port, username, password)
                
            self.connections[session_id] = ssh
            return f"SSH连接成功: {username}@{hostname}:{port}", session_id
        except Exception as e:
            return f"连接失败: {str(e)}"
    

    async def exec_command(self, command: str, session_id: str) -> str:
        """通过持久连接执行命令"""
        
        print(self.connections.keys())
        print("session_id:", session_id)
        if session_id not in self.connections:
            return "无活动SSH连接，请先建立连接"
        
        ssh = self.connections[session_id]
        try:
            stdin, stdout, stderr = ssh.exec_command(command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            return f"$ {command}\n{output}{error}" if output or error else "命令已执行"
        except Exception as e:
            return f"执行失败: {str(e)}"
    

    async def close_connection(self, session_id: str) -> str:
        """关闭当前会话的SSH连接"""
        # client_id = ctx.session_id
        if session_id in self.connections:
            ssh = self.connections.pop(session_id)
            ssh.close()
            return "SSH连接已关闭"
        return "无活动连接可关闭"

# 初始化连接管理器
ssh_manager = SSHConnectionManager()

# 注册MCP工具
@mcp.tool()
async def ssh_connect(hostname: str, port: int, username: str, 
                      password: str = None, key_path: str = None) -> Dict[str, Any]:
    """
    建立持久SSH连接
    参数:
      hostname: 服务器地址
      port: SSH端口
      username: 用户名
      password: 密码(可选)
      key_path: SSH密钥路径(可选)
    """
    result, session_id = await ssh_manager.create_connection(hostname, port, username, password, key_path)
    return {"session_id": session_id, "message": result}

@mcp.tool()
async def ssh_exec(command: str, session_id: str):
    """
    在SSH连接上执行命令
    参数:
      command: 要执行的Shell命令
    """
    result = await ssh_manager.exec_command(command, session_id)
    return {
        "message": result
    }

@mcp.tool()
async def ssh_disconnect(session_id):
    """关闭当前SSH连接"""
    result = await ssh_manager.close_connection(session_id)
    return {
        "message": result
    }

@mcp.tool()
async def ssh_connect_by_alias(host_alias: str):
    """
    通过 ~/.ssh/config 中的别名建立SSH连接
    参数:
      host_alias: SSH配置中的主机别名（如 sugon）
    """
    try:
        # 1. 加载SSH配置
        config = SSHConnectionManager.load_ssh_config(host_alias)
        
        # 2. 提取关键参数（带默认值处理）
        hostname = config.get("hostname", "")
        port = int(config.get("port", 22))
        username = config.get("user", "")
        identity_file = config.get("identityfile", [None])[0]  # 可能有多密钥
        
        print(f"Connecting to {username}@{hostname}:{port} using key {identity_file}")
        # print(ctx.request_id, " ", ctx.client_id, " ", ctx.session.id)

        # 3. 建立连接
        result, session_id = await ssh_manager.create_connection(
            hostname=hostname,
            port=port,
            username=username,
            key_path=identity_file  # 自动使用配置的密钥
        )
        print(f"Connection result: {result}")
        return {
            "session_id": session_id, "message": result
        }
    
    except Exception as e:
        print(f"Error connecting to {host_alias}: {str(e)}")
        return TextContent(text=f"连接失败: {str(e)}")

# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run(transport="streamable-http", mount_path="/mcp")

if __name__ == "__main__":
    main()