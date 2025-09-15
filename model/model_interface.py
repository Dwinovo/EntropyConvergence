"""
模型接口模块，提供语言模型的抽象基类。
通过将模型相关功能组合在一起，确保高内聚性。
"""
from abc import ABC, abstractmethod
from typing import Optional


class ModelInterface(ABC):
    """语言模型实现的抽象基类。"""
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型和分词器。"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, max_length: Optional[int] = None) -> str:
        """根据给定的提示生成响应。
        
        Args:
            prompt: 输入提示文本
            max_length: 生成文本的最大长度（可选）
            
        Returns:
            生成的响应文本
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载并准备就绪。
        
        Returns:
            如果模型已加载返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型以释放内存。"""
        pass
