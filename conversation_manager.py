"""
对话管理器，用于协调模型间的问答会话。
高内聚：管理对话流程和状态。
低耦合：使用依赖注入来处理模型。
"""
import json
import os
import math
from datetime import datetime
from typing import List, Dict, Optional

from model.aihubmix_gemini import RemoteModel
from model.local_model import LocalModel
# 移除接口依赖，直接使用鸭子类型


class ConversationRound:
    """表示单轮对话的数据类。"""
    
    def __init__(self, round_number: int, question: str = "", answer: str = "", conditional_entropy_bits_per_token: float = None, perplexity: float = None):
        """
        初始化对话轮次。
        
        Args:
            round_number: 轮次编号
            question: 问题内容
            answer: 答案内容
        """
        self.round_number = round_number  # 轮次编号
        self.question = question  # 问题内容
        self.answer = answer  # 答案内容
        self.conditional_entropy_bits_per_token = conditional_entropy_bits_per_token  # 本轮答案逐token条件熵的平均值（bits/token）
        self.perplexity = perplexity  # 困惑度（PPL） = 2^H
    
    def __str__(self) -> str:
        """返回格式化的对话轮次字符串。"""
        return f"第 {self.round_number} 轮:\n问: {self.question}\n答: {self.answer}\n"
    
    def to_dict(self) -> Dict:
        """将对话轮次转换为字典格式。"""
        return {
            "round_number": self.round_number,
            "question": self.question,
            "answer": self.answer,
            "conditional_entropy_bits_per_token": self.conditional_entropy_bits_per_token,
            "perplexity": self.perplexity
        }
    


class ConversationManager:
    """管理问答模型之间的对话流程。"""
    
    def __init__(self, question_model:RemoteModel, answer_model:LocalModel):
        """
        初始化对话管理器。
        
        Args:
            question_model: 问题生成模型
            answer_model: 答案生成模型
        """
        self.question_model = question_model  # 问题生成模型
        self.answer_model = answer_model  # 答案生成模型
        self.conversation_history: List[ConversationRound] = []  # 对话历史记录
        self.topic = ""  # 主题
        self.average_conditional_entropy_bits_per_token: Optional[float] = None  # 本次对话的平均条件熵（bits/token）
    
    def set_topic(self, topic: str) -> None:
        """
        设置对话的主题。
        
        Args:
            topic: 主题内容
        """
        self.topic = topic
        

    def start_conversation_from_topic(self, topic: str, num_rounds: int = 5) -> List[ConversationRound]:
        """
        从主题开始对话：先把 topic 交给提问模型生成第1个问题，然后本地回答模型在回答前测熵。
        后续轮次基于对话历史与 topic 继续提问与回答。
        
        Args:
            topic: 主题内容（来自 question/ai.txt）
            num_rounds: 对话轮数
        
        Returns:
            对话轮次列表
        """
        print(f"开始 {num_rounds} 轮对话（从主题开始）...")
        self.conversation_history = []
        self.topic = topic

        current_question = None
        for round_num in range(1, num_rounds + 1):
            print(f"\n--- 第 {round_num} 轮 ---")
            # 生成问题
            if round_num == 1:
                current_question= self.question_model.generate_response(topic)
            else:
                context_for_next_question = self._build_context_for_next_round(round_num - 1)
                current_question= self.question_model.generate_response(context_for_next_question)
            
            print(f"问题: {current_question}")
            
            # 生成答案并计算条件熵
            current_answer, conditional_entropy = self.answer_model.generate_response(current_question)
            perplexity = 2 ** conditional_entropy  # 困惑度 = 2^H
            
            print(f"答案: {current_answer}")
            print(f"条件熵: {conditional_entropy:.4f} bits/token")
            print(f"困惑度: {perplexity:.4f}")
            
            # 创建对话轮次对象
            round_obj = ConversationRound(
                round_number=round_num,
                question=current_question,
                answer=current_answer,
                conditional_entropy_bits_per_token=conditional_entropy,
                perplexity=perplexity
            )
            
            # 添加到对话历史
            self.conversation_history.append(round_obj)

        # 计算整个对话的平均条件熵
        if self.conversation_history:
            total_entropy = sum(round_obj.conditional_entropy_bits_per_token for round_obj in self.conversation_history)
            self.average_conditional_entropy_bits_per_token = total_entropy / len(self.conversation_history)
            print(f"\n整个对话的平均条件熵: {self.average_conditional_entropy_bits_per_token:.4f} bits/token")

        return self.conversation_history

    
    def _build_context_for_next_round(self, current_round: int) -> str:
        """
        为下一轮构建包含对话历史的文本格式上下文。
        
        Args:
            current_round: 当前轮次编号
            
        Returns:
            构建好的文本格式上下文字符串
        """
        context_parts = []
        if self.topic:
            context_parts.append(f"topic: {self.topic}")
        
        # 添加之前的问答对，使用question/answer格式
        for round_obj in self.conversation_history[:current_round]:
            context_parts.append(f"question: {round_obj.question}")
            context_parts.append(f"answer: {round_obj.answer}")
        
        return "\n\n".join(context_parts)
    
    
    def save_conversation_to_json(self, filename: str) -> None:
        """
        将对话保存到JSON文件。
        
        Args:
            filename: 保存的JSON文件名
        """
        conversation_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_rounds": len(self.conversation_history),
                "average_conditional_entropy_bits_per_token": self.average_conditional_entropy_bits_per_token
            },
            "topic": self.topic,
            "conversation_rounds": [round_obj.to_dict() for round_obj in self.conversation_history]
        }
        
        # 确保日志目录存在
        os.makedirs("log", exist_ok=True)
        with open(f"log/{filename}", 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到JSON文件: {filename}")

    
    def get_latest_round(self) -> Optional[ConversationRound]:
        """
        获取最近的对话轮次。
        
        Returns:
            最近的对话轮次对象，如果没有则返回None
        """
        return self.conversation_history[-1] if self.conversation_history else None
    
    def get_round_count(self) -> int:
        """
        获取已完成的轮次数量。
        
        Returns:
            已完成的轮次数
        """
        return len(self.conversation_history)
