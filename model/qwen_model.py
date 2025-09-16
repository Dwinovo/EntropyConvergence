"""
Qwen2-7B模型实现，用于生成答案。
高内聚：包含所有Qwen特定的功能。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import math
from .model_interface import ModelInterface


class QwenAnswerModel(ModelInterface):
    """用于回答问题的Qwen2-7B模型。"""
    
    def __init__(self, model_path: str):
        """初始化Qwen答案生成模型。
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path  # 模型路径
        self.model = None  # 模型对象，初始为None
        self.tokenizer = None  # 分词器对象，初始为None
        self.device = "cuda"
    
    def load_model(self) -> None:
        """加载Qwen2-7B模型和分词器。"""
        print(f"正在从 {self.model_path} 加载Qwen2-7B模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True  # 信任远程代码
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,  # 使用半精度浮点数以节省内存
            device_map="cuda"
        )
        self.model.eval()
        torch.manual_seed(42)
        print("Qwen2-7B模型加载成功！")
    
    def generate_response(self, prompt: str, max_length: Optional[int] = 512) -> str:
        """基于给定的问题生成答案。
        
        Args:
            prompt: 输入的问题
            max_length: 生成文本的最大长度
            
        Returns:
            生成的答案文本
            
        Raises:
            RuntimeError: 如果模型未加载
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model() 方法。")
        
        # 为答案生成格式化提示词
        formatted_prompt = f"""
You are a knowledgeable assistant. Given the context below, provide a concise and accurate answer.
Context:
{prompt}
Please provide one concise, accurate answer:
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
                pad_token_id=self.tokenizer.pad_token_id,  # 填充token ID
                eos_token_id=self.tokenizer.eos_token_id  # 结束token ID
            )
        
        # 仅解码新生成的token（答案部分）
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # 跳过输入部分
            skip_special_tokens=True  # 跳过特殊token
        ).strip()  # 去除首尾空白
        
        # 移除可能残留的特殊token
        response = response.replace("<|im_end|>", "").strip()
        
        return response

    def compute_entropy_for_question(self, prompt: str) -> float:
        """计算给定问题在当前上下文下的瞬时预测熵（比特）。
        
        熵定义为对下一token分布的香农熵：H = -Σ p(x) log2 p(x)
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model() 方法。")

        formatted_prompt = f"""
You are a knowledgeable assistant. Given the context below, provide a concise and accurate answer.
Context:
{prompt}
Please provide one concise, accurate answer:
"""

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
            # 取最后一个位置的logits，并转为float32以避免FP16数值不稳定
            last_logits = outputs.logits[0, -1, :].to(dtype=torch.float32)
            # 使用数值稳定的log_softmax
            log_probs = torch.log_softmax(last_logits, dim=-1)  # 自然对数
            probs = torch.exp(log_probs)
            entropy_nats = -(probs * log_probs).sum()
            entropy_bits = (entropy_nats / math.log(2)).item()

            # 兜底：如果出现非有限结果，做一次中心化后重算
            if not math.isfinite(entropy_bits):
                last_logits = (last_logits - last_logits.max()).to(dtype=torch.float32)
                log_probs = torch.log_softmax(last_logits, dim=-1)
                probs = torch.exp(log_probs)
                entropy_nats = -(probs * log_probs).sum()
                entropy_bits = (entropy_nats / math.log(2)).item()

        return float(entropy_bits)
    
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
        print("Qwen2-7B模型已卸载。")
