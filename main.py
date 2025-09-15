"""
主脚本：使用 AIHubMix(Gemini) 作为提问模型 + 本地回答模型连续对话。
流程：读取 question/ai.txt 作为 topic -> 提问模型生成问题 -> 回答前测熵 -> 继续多轮。
低耦合：依赖注入 + 组合模式。
"""
import os
import sys
from datetime import datetime
from model.aihubmix_gemini import AIHubMixQuestionModel
from model.qwen_model import QwenAnswerModel
from conversation_manager import ConversationManager
from dotenv import load_dotenv
load_dotenv()
def main():
    """运行对话系统的主函数。"""
    
    # 回答模型路径配置（仅保留本地回答模型）
    qwen_path = "/root/autodl-fs/Qwen2-7B-Instruct"

    # 验证回答模型路径是否存在
    if not os.path.exists(qwen_path):
        print(f"错误：找不到Qwen模型路径: {qwen_path}")
        sys.exit(1)
    
    print("=== AI对话系统 ===")
    print("AIHubMix(Gemini 提问) + Qwen2-7B 本地回答（回答前测熵）")
    print("=" * 50)
    
    try:
        # 初始化模型
        question_model = AIHubMixQuestionModel()
        answer_model = QwenAnswerModel(qwen_path)

        # 加载模型
        print("\n2. 正在加载模型...")
        question_model.load_model()
        answer_model.load_model()
        conversation_manager = ConversationManager(question_model, answer_model)

        # 读取主题文件
        topic_file = "topic/ai.txt"
        if not os.path.exists(topic_file):
            print(f"错误：找不到主题文件: {topic_file}")
            return
        with open(topic_file, 'r', encoding='utf-8') as f:
            topic = f.read().strip()
        if not topic:
            print("错误：主题文件为空。")
            return

        print("\n5. 开始连续对话（从主题生成首问）...")
        print("=" * 50)
        rounds = conversation_manager.start_conversation_from_topic(topic, num_rounds=10)
        
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"conversation_{timestamp}.json"
        # 保存JSON格式
        conversation_manager.save_conversation_to_json(json_filename)
        # 绘制并保存熵-轮数折线图
        image_path = f"img/{json_filename.replace('.json', '.png')}"
        conversation_manager.save_entropy_plot(image_path)
        
        print(f"JSON格式已保存到: {json_filename}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理模型
        print("\n6. 正在清理资源...")
        try:
            if 'question_model' in locals():
                question_model.unload_model()
            if 'answer_model' in locals():
                answer_model.unload_model()
            print("模型已成功卸载。")
        except Exception as e:
            print(f"清理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
