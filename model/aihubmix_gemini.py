"""
AIHubMix (推理时代) 问题模型实现，通过 OpenAI 兼容的 /v1/chat/completions 接口调用（用于 Gemini 等）。
高内聚：封装 API 细节，仅暴露 ModelInterface 能力。
低耦合：通过依赖注入与对话管理协作。

参考文档：见 https://docs.aihubmix.com/cn （OpenAI 兼容，替换 base_url 与 key）
"""
import os
import json
from typing import Optional
import requests
from .model_interface import ModelInterface


class AIHubMixQuestionModel(ModelInterface):
    """使用 AIHubMix 提问（Gemini 作为问题生成模型）。"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None, system_prompt: Optional[str] = None):
        """初始化 API 客户端配置。

        Args:
            base_url: AIHubMix 基础地址，示例：https://aihubmix.com 或 https://api.aihubmix.com
            api_key: AIHubMix API Key，形如 sk-***
            model_name: 具体模型名称（例如 Gemini 系列）。必须显式提供，避免臆测。
            system_prompt: 系统提示词，可选
        """
        self.base_url = (base_url or os.getenv("AIHUBMIX_BASE_URL") or "https://aihubmix.com").rstrip("/")
        self.api_key = api_key or os.getenv("AIHUBMIX_API_KEY")
        self.model_name = model_name or os.getenv("AIHUBMIX_MODEL")
        self.system_prompt = system_prompt or (
            "You are a skilled question generator. Given the topic and prior Q&A context, ask ONE concise, insightful question that advances the discussion. Output only the question."
        )
        self._is_ready = False

    def load_model(self) -> None:
        """校验必要配置，AI 接口无需本地加载大模型。"""
        if not self.api_key:
            raise RuntimeError("缺少 AIHubMix API Key，请设置环境变量 AIHUBMIX_API_KEY。")
        if not self.model_name:
            raise RuntimeError("缺少模型名称，请设置环境变量 AIHUBMIX_MODEL（例如 Gemini 模型标识）。")
        # 进行一次轻量级健康检查（可选）
        self._is_ready = True

    def generate_response(self, prompt: str, max_length: Optional[int] = 256) -> str:
        """调用 AIHubMix 的 Chat Completions 生成问题。"""
        if not self.is_loaded():
            raise RuntimeError("问题模型未就绪，请先调用 load_model()。")

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "stream": False,
            "max_tokens": max_length or 256,
        }
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        except Exception as req_err:
            raise RuntimeError(f"请求 AIHubMix 失败: {req_err}") from req_err

        if resp.status_code != 200:
            raise RuntimeError(f"AIHubMix 返回错误状态码 {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
            # OpenAI 兼容返回格式：choices[0].message.content
            content = data["choices"][0]["message"]["content"].strip()
        except Exception as parse_err:
            raise RuntimeError(f"解析 AIHubMix 响应失败: {parse_err}; 原始: {resp.text}") from parse_err

        return content

    def is_loaded(self) -> bool:
        return self._is_ready

    def unload_model(self) -> None:
        # 无持久资源需要释放，但提供一致接口
        self._is_ready = False
