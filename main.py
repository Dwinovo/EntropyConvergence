
import os
import sys
import argparse
from datetime import datetime
from model.aihubmix_gemini import AIHubMixQuestionModel
from model.qwen_model import QwenAnswerModel
from conversation_manager import ConversationManager
from dotenv import load_dotenv
load_dotenv()
def parse_args():
    parser = argparse.ArgumentParser(description="AI 对话系统 - 提问模型(Gemini) + 回答模型(Qwen/Llama等)")
    parser.add_argument("--model_path", type=str, help="回答模型的路径，例如 /root/autodl-fs/Qwen2-7B-Instruct", default="/root/autodl-fs/Qwen2-7B-Instruct")
    parser.add_argument("--topic", type=str, help="topic 名称（位于 topic/ 下，支持 ai 或 ai.txt）", default="art.txt")
    parser.add_argument("--rounds", type=int, default=20, help="对话轮数，默认20")
    return parser.parse_args()

def main():
    """运行对话系统的主函数。"""
    
    print("=== AI对话系统 ===")
    print("AIHubMix(Gemini 提问) + Qwen2-7B 本地回答（回答前测熵）")
    print("=" * 50)
    
    try:
        # 解析命令行参数
        args = parse_args()
        model_path = args.model_path
        # 规范化 topic：只需名称，自动从 topic/ 读取，允许不带 .txt
        topic_arg = args.topic.strip()
        # 若用户误传入路径，取其文件名
        topic_name = os.path.basename(os.path.normpath(topic_arg)) if topic_arg else topic_arg
        # 若无扩展名则补 .txt
        if topic_name and not os.path.splitext(topic_name)[1]:
            topic_name += ".txt"
        topic_file = os.path.join("topic", topic_name)
        num_rounds = int(args.rounds)

        # 初始化模型
        question_model = AIHubMixQuestionModel()
        answer_model = QwenAnswerModel(model_path)

        # 加载模型
        print("\n2. 正在加载模型...")
        question_model.load_model()
        answer_model.load_model()
        conversation_manager = ConversationManager(question_model, answer_model)

        # 读取主题文件
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
        rounds = conversation_manager.start_conversation_from_topic(topic, num_rounds=num_rounds)
        
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 取模型名与topic文件名
        model_name = os.path.basename(os.path.normpath(model_path))
        topic_name = os.path.basename(os.path.normpath(topic_file))
        # 去除topic文件的扩展名
        topic_stem = os.path.splitext(topic_name)[0]
        base_name = f"{model_name}-{topic_stem}-{num_rounds}"
        json_filename = f"{base_name}.json"
        # 保存JSON格式
        conversation_manager.save_conversation_to_json(json_filename)
        # 绘制并保存综合指标图（熵+PPL 同图）
        image_path_metrics = f"img/{base_name}_metrics.png"
        conversation_manager.save_metrics_plot(image_path_metrics)
        
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
