
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, List
import math
import os


class LocalModel:
    """统一的本地语言模型，支持Qwen、Llama等所有transformers兼容的模型。"""
    
    def __init__(self, model_path: str):
        """初始化本地模型。
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model: AutoModelForCausalLM = None
        self.tokenizer: AutoTokenizer = None
    
    def load_model(self) -> None:
        """加载模型和分词器。"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # 仅使用CUDA，若不可用则报错
        if not torch.cuda.is_available():
            raise RuntimeError("需要可用的CUDA设备，但未检测到GPU。")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model = self.model.half().to("cuda")
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.manual_seed(42)

    def generate_response(self, prompt: str, max_length: Optional[int] = 512) -> Tuple[str, float]:
        """生成响应并返回逐token条件熵。
        
        Args:
            prompt: 输入提示
            max_length: 生成文本的最大长度
            
        Returns:
            生成的响应文本和平均逐token条件熵（bits/token，float）
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model() 方法。")
        
        # 对输入进行编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to("cuda")
        
        
        # 生成响应，获取每步的logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                top_k=50,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # 解码新生成的tokens
        sequences = outputs.sequences
        scores = outputs.scores or []
        generated_ids = sequences[0][inputs["input_ids"].shape[-1]:]
        # 解码需要CPU上的整数列表
        response_tokens = generated_ids.detach().cpu().tolist()
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        response = response.replace("<|im_end|>", "").strip()
        
        # 计算逐token条件熵
        per_token_entropy = []
        for step_scores in scores:
            entropy = self.calculate_entropy_from_logits(step_scores[0])
            per_token_entropy.append(float(entropy))
        # 返回平均条件熵（bits/token），与对话管理器的用法一致
        avg_entropy_bits = float(sum(per_token_entropy) / max(1, len(per_token_entropy)))
        return response, avg_entropy_bits
    
    @staticmethod
    def calculate_entropy_from_logits(logits: torch.Tensor) -> float:
        """从logits计算香农熵。
        
        Args:
            logits: 模型输出的logits张量，形状为 [vocab_size]
            
        Returns:
            香农熵值（bits）
        """
        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        return entropy / math.log(2)
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self.model is not None and self.tokenizer is not None
    
    def unload_model(self) -> None:
        """卸载模型以释放内存。"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        print("本地模型已卸载。")

