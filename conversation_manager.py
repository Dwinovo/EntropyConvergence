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
from model.model_interface import ModelInterface


class ConversationRound:
    """表示单轮对话的数据类。"""
    
    def __init__(self, round_number: int, question: str = "", answer: str = "", context_entropy_bits: float = None):
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
        self.context_entropy_bits = context_entropy_bits  # 回答前的瞬时熵（比特）
    
    def __str__(self) -> str:
        """返回格式化的对话轮次字符串。"""
        return f"第 {self.round_number} 轮:\n问: {self.question}\n答: {self.answer}\n"
    
    def to_dict(self) -> Dict:
        """将对话轮次转换为字典格式。"""
        return {
            "round_number": self.round_number,
            "question": self.question,
            "answer": self.answer,
            "context_entropy_bits": self.context_entropy_bits
        }
    


class ConversationManager:
    """管理问答模型之间的对话流程。"""
    
    def __init__(self, question_model: ModelInterface, answer_model: ModelInterface):
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
                current_question = self.question_model.generate_response(topic)
            else:
                context_for_next_question = self._build_context_for_next_round(round_num - 1)
                current_question = self.question_model.generate_response(context_for_next_question)

            # 回答前计算瞬时熵
            context_entropy_bits = None
            if hasattr(self.answer_model, "compute_entropy_for_question"):
                try:
                    context_entropy_bits = self.answer_model.compute_entropy_for_question(current_question)
                except Exception:
                    context_entropy_bits = None

            # 生成答案
            answer = self.answer_model.generate_response(current_question)
            round_obj = ConversationRound(round_num, current_question, answer, context_entropy_bits)
            self.conversation_history.append(round_obj)

            entropy_str = f"{context_entropy_bits:.4f} bits" if context_entropy_bits is not None else "NA"
            print(f"问题: {current_question}\n答案: {answer}\n熵: {entropy_str}")

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
                "total_rounds": len(self.conversation_history)
            },
            "topic": self.topic,
            "conversation_rounds": [round_obj.to_dict() for round_obj in self.conversation_history]
        }
        
        with open(f"log/{filename}", 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到JSON文件: {filename}")

    def save_entropy_plot(self, image_path: str) -> None:
        """
        基于对话历史绘制“熵-轮数”折线图，并保存到指定路径。
        
        Args:
            image_path: 输出图像的完整路径，例如 img/conversation_XXXX.png
        """
        # 延迟导入matplotlib，避免环境缺少时影响核心流程
        try:
            import matplotlib
            matplotlib.use('Agg')  # 无显示环境下渲染
            import matplotlib.pyplot as plt
        except Exception as import_err:
            print(f"无法绘制熵曲线（matplotlib不可用）: {import_err}")
            return

        if not self.conversation_history:
            print("无对话历史，跳过熵曲线绘制。")
            return

        # 构造数据
        x_rounds = [r.round_number for r in self.conversation_history]
        y_entropy = [
            (r.context_entropy_bits if (r.context_entropy_bits is not None and math.isfinite(r.context_entropy_bits)) else float('nan'))
            for r in self.conversation_history
        ]

        # 确保目录存在
        out_dir = os.path.dirname(image_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # 绘图
        plt.figure(figsize=(8, 4.5))
        plt.plot(x_rounds, y_entropy, marker='o', linestyle='-', color='#1f77b4')
        plt.title('瞬时熵随轮数变化')
        plt.xlabel('轮数')
        plt.ylabel('熵 (bits)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(x_rounds)
        plt.tight_layout()
        plt.savefig(image_path, dpi=150)
        plt.close()
        print(f"熵曲线已保存到: {image_path}")
    
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
