"""
yaml
k: v
"""
import os

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

# 加载项目根目录 .env（不存在时静默跳过；不覆盖已存在的环境变量）。
# Chainlit / Streamlit 启动时也会自动加载 .env，但命令行脚本（训练、评估等）不会，
# 因此在配置层统一加载，保证任何入口都能读取 DASHSCOPE_API_KEY 等配置。
load_dotenv(get_abs_path(".env"))


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
