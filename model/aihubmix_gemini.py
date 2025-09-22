import os
from typing import Optional, List, Tuple
from openai import OpenAI

class RemoteModel:

    def __init__(self,  model_name: str):

        self.base_url = "https://aihubmix.com/v1"
        self.api_key = os.getenv("AIHUBMIX_API_KEY")
        self.model_name = model_name
        self.system_prompt = (
            "You are a skilled questioner. Based on the given topic and the context of the Q&A, ask a concise, insightful, and in-depth question that revolves around the topic and context to push the discussion forward. Note: Only output the question itself, and it must be centered on the topic."
        )
        self._is_ready = False
        self.client:OpenAI = None

    def load_model(self) -> None:
        if not self.api_key:
            raise RuntimeError("缺少 AIHubMix API Key，请设置环境变量 AIHUBMIX_API_KEY。")

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
