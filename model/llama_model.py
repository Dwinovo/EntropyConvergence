"""
Llama3-8B模型实现，用于生成问题。
高内聚：包含所有Llama特定的功能。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from .model_interface import ModelInterface


class LlamaQuestionModel(ModelInterface):
    """基于上下文生成问题的Llama3-8B模型。"""
    
    def __init__(self, model_path: str):
        """初始化Llama问题生成模型。
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path  # 模型路径
        self.model = None  # 模型对象，初始为None
        self.tokenizer = None  # 分词器对象，初始为None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设备选择：优先使用GPU
    
    def load_model(self) -> None:
        """加载Llama3-8B模型和分词器。"""
        print(f"正在从 {self.model_path} 加载Llama3-8B模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 如果没有pad_token，则使用eos_token作为pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,  # 使用半精度浮点数以节省内存
            device_map="cuda"
        )
        
        print("Llama3-8B模型加载成功！")
    
    def generate_response(self, prompt: str, max_length: Optional[int] = 512) -> str:
        """基于给定的上下文/提示生成问题。
        
        Args:
            prompt: 输入的上下文或提示
            max_length: 生成文本的最大长度
            
        Returns:
            生成的问题文本
            
        Raises:
            RuntimeError: 如果模型未加载
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model() 方法。")
        
        # 为问题生成格式化提示词
        formatted_prompt = f"""
You are an fucking intelligent assistant. Based on the given context and topic, ask one concise question that deepens the exploration of the topic.
Conversation history: {prompt}
Please output only the question and nothing fucking else or i will kill you
"""
        
        # 对输入进行编码
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",  # 返回PyTorch张量
            truncation=True,  # 启用截断
            max_length=2048  # 最大输入长度
        ).to(self.device)
        
        # 生成响应（不计算梯度以节省内存）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # 最大新生成token数
                temperature=0.7,  # 控制生成的随机性
                pad_token_id=self.tokenizer.eos_token_id,  # 填充token ID
                eos_token_id=self.tokenizer.eos_token_id  # 结束token ID
            )
        
        # 仅解码新生成的token（问题部分）
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # 跳过输入部分
            skip_special_tokens=True  # 跳过特殊token
        ).strip()  # 去除首尾空白
        
        return response
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载。
        
        Returns:
            如果模型和分词器都已加载返回True，否则返回False
        """
        return self.model is not None and self.tokenizer is not None
    
    def unload_model(self) -> None:
        """卸载模型以释放内存。"""
        # 删除模型对象
        if self.model is not None:
            del self.model
            self.model = None
        # 删除分词器对象
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        print("Llama3-8B模型已卸载。")
