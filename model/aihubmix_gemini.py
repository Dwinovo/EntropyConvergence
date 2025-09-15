"""
AIHubMix (推理时代) 问题模型实现，通过 OpenAI 兼容的 /v1/chat/completions 接口调用（用于 Gemini 等）。
高内聚：封装 API 细节，仅暴露 ModelInterface 能力。
低耦合：通过依赖注入与对话管理协作。

参考文档：见 https://docs.aihubmix.com/cn/api/Aihubmix-Integration （OpenAI 兼容，设置 base_url 与 api_key）
"""
import os
from typing import Optional
from openai import OpenAI
from .model_interface import ModelInterface


class AIHubMixQuestionModel(ModelInterface):
    """使用 AIHubMix 提问（Gemini 作为问题生成模型）。"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None, system_prompt: Optional[str] = None):
        """初始化 API 客户端配置。

        Args:
            base_url: AIHubMix 基础地址，应包含 /v1（示例：https://aihubmix.com/v1 或 https://api.aihubmix.com/v1）
            api_key: AIHubMix API Key，形如 sk-***
            model_name: 具体模型名称（例如 Gemini 系列）。必须显式提供，避免臆测。
            system_prompt: 系统提示词，可选
        """
        self.base_url = (base_url or os.getenv("AIHUBMIX_BASE_URL") or "https://aihubmix.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("AIHUBMIX_API_KEY")
        self.model_name = model_name or os.getenv("AIHUBMIX_MODEL")
        self.system_prompt = system_prompt or (
            "You are a skilled question generator. Given the topic and prior Q&A context, ask ONE concise, insightful question that advances the discussion. Output only the question."
        )
        self._is_ready = False
        self.client: Optional[OpenAI] = None

    def load_model(self) -> None:
        """校验必要配置并初始化 OpenAI 客户端（AIHubMix 转发）。"""
        if not self.api_key:
            raise RuntimeError("缺少 AIHubMix API Key，请设置环境变量 AIHUBMIX_API_KEY。")
        if not self.model_name:
            raise RuntimeError("缺少模型名称，请设置环境变量 AIHUBMIX_MODEL（例如 Gemini 模型标识）。")
        # 初始化 OpenAI 客户端（兼容 AIHubMix）
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._is_ready = True

    def generate_response(self, prompt: str, max_length: Optional[int] = 256) -> str:
        """调用 AIHubMix 的 Chat Completions 生成问题。"""
        if not self.is_loaded() or self.client is None:
            raise RuntimeError("问题模型未就绪，请先调用 load_model()。")

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        try:
            
            content = (resp.choices[0].message.content)
        except Exception as parse_err:
            raise RuntimeError(f"解析 AIHubMix(OpenAI SDK) 响应失败: {parse_err}") from parse_err

        return content

    def is_loaded(self) -> bool:
        return self._is_ready

    def unload_model(self) -> None:
        # 无持久资源需要释放，但提供一致接口
        self._is_ready = False
        self.client = None
