
import os
import sys
import argparse
from datetime import datetime
from model.aihubmix_gemini import RemoteModel
from model.local_model import LocalModel
from conversation_manager import ConversationManager
from dotenv import load_dotenv
load_dotenv()
def parse_args():
    parser = argparse.ArgumentParser(description="AI 对话系统 - 提问模型(Gemini) + 回答模型(Qwen/Llama等)")
    # /root/autodl-fs/Meta-Llama-3-8B-Instruct
    parser.add_argument("--model_path", type=str, help="回答模型的路径，例如 /root/autodl-fs/Qwen2-7B-Instruct", default="/root/autodl-fs/Qwen2-7B-Instruct")
    parser.add_argument("--topic", type=str, help="topic 名称（位于 topic/ 下，支持 ai 或 ai.txt）", default="ai.txt")
    parser.add_argument("--rounds", type=int, default=20, help="对话轮数，默认20")
    parser.add_argument("--name", type=str, default="", help="额外的命名后缀（可选），会拼接到文件名末尾")
    return parser.parse_args()

def main():
    """运行对话系统的主函数。"""
    
    try:
        # 解析命令行参数
        args = parse_args()
        model_path = args.model_path
        # 规范化 topic：只需名称，自动从 topic/ 读取，允许不带 .txt
        topic_arg = args.topic
        
        
        num_rounds = int(args.rounds)
        # 规范化 name：可为空；若包含路径仅取最后一段；空格替换为下划线
        name_arg = (args.name or "").strip()
        name_part = os.path.basename(os.path.normpath(name_arg)) if name_arg else ""
        name_part = name_part.replace(" ", "_") if name_part else ""
        # 初始化模型
        question_model = RemoteModel("gemini-2.5-flash")
        answer_model = LocalModel(model_path)

        # 加载模型
        print("\n2. 正在加载模型...")
        question_model.load_model()
        answer_model.load_model()
        conversation_manager = ConversationManager(question_model, answer_model)

        # 解析主题文件为绝对路径：
        # 支持以下输入：绝对路径 X / X.txt；或名称/相对路径 ai、ai.txt、topic/ai、topic/ai.txt
        project_root = os.path.dirname(os.path.abspath(__file__))
        topic_candidates = []
        if os.path.isabs(topic_arg):
            topic_candidates.append(topic_arg)
            if not topic_arg.endswith('.txt'):
                topic_candidates.append(f"{topic_arg}.txt")
        else:
            topic_dir_abs = os.path.join(project_root, "topic")
            # 相对当前工作目录与项目根目录、以及 topic 目录的候选
            rel_variants = [topic_arg]
            if not topic_arg.endswith('.txt'):
                rel_variants.append(f"{topic_arg}.txt")
            topic_candidates.extend(rel_variants)
            topic_candidates.extend([
                os.path.join('topic', v) for v in rel_variants
            ])
            topic_candidates.extend([
                os.path.join(topic_dir_abs, v) for v in rel_variants
            ])

        topic_file = next((p for p in topic_candidates if os.path.exists(p)), None)
        if topic_file is None:
            print(f"错误：找不到主题文件。已尝试: {', '.join(topic_candidates)}")
            return
        topic_file = os.path.abspath(topic_file)
        # 读取主题文件
        with open(topic_file, 'r', encoding='utf-8') as f:
            topic = f.read().strip()
        if not topic:
            print("错误：主题文件为空。")
            return

        print("\n5. 开始连续对话（从主题生成首问）...")
        print("=" * 50)
        rounds = conversation_manager.start_conversation_from_topic(topic, num_rounds=num_rounds)
        
        # 取模型名与topic文件名
        model_name = os.path.basename(os.path.normpath(model_path))
        topic_name = os.path.basename(os.path.normpath(topic_file))
        # 去除topic文件的扩展名
        topic_stem = os.path.splitext(topic_name)[0]
        base_name = f"{model_name}-{topic_stem}-{num_rounds}"
        if name_part:
            base_name = f"{base_name}-{name_part}"
        json_filename = f"{base_name}.json"
        # 保存JSON格式
        conversation_manager.save_conversation_to_json(json_filename)

        
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
