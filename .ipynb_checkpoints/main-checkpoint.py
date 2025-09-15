"""
主脚本，用于协调Llama3-8B和Qwen2-7B之间的5轮对话。
低耦合：使用依赖注入和组合模式。
"""
import os
import sys
from datetime import datetime
from llama_model import LlamaQuestionModel
from qwen_model import QwenAnswerModel
from conversation_manager import ConversationManager


def main():
    """运行对话系统的主函数。"""
    
    # 模型路径配置
    llama_path = "/root/autodl-fs/Meta-Llama-3-8B-Instruct"
    qwen_path = "/root/autodl-fs/Qwen2-7B-Instruct"
    
    # 验证模型路径是否存在
    if not os.path.exists(llama_path):
        print(f"错误：找不到Llama模型路径: {llama_path}")
        sys.exit(1)
    
    if not os.path.exists(qwen_path):
        print(f"错误：找不到Qwen模型路径: {qwen_path}")
        sys.exit(1)
    
    print("=== AI对话系统 ===")
    print("Llama3-8B（问题生成器）+ Qwen2-7B（答案生成器）")
    print("=" * 50)
    
    try:
        # 初始化模型
        print("\n1. 正在初始化模型...")
        llama_model = LlamaQuestionModel(llama_path)
        qwen_model = QwenAnswerModel(qwen_path)
        
        # 加载模型
        print("\n2. 正在加载模型...")
        llama_model.load_model()
        qwen_model.load_model()
        
        # 初始化对话管理器
        print("\n3. 正在设置对话管理器...")
        conversation_manager = ConversationManager(llama_model, qwen_model)
        
        # 从用户获取初始上下文
        print("\n4. 请提供对话的初始上下文：")
        print("（此上下文将用于生成第一个问题）")
        
        # 非交互式运行时的默认上下文
        default_context = """
        人工智能技术正在快速发展，特别是大语言模型在各个领域都展现出了强大的能力。
        从自然语言处理到代码生成，从创意写作到科学研究，AI正在改变我们的工作和生活方式。
        同时，这也带来了一些挑战，包括伦理问题、就业影响、数据隐私等。
        """
        
        try:
            context = input().strip()
            if not context:
                print("未提供上下文，使用默认上下文...")
                context = default_context.strip()
        except (EOFError, KeyboardInterrupt):
            print("使用默认上下文...")
            context = default_context.strip()
        
        conversation_manager.set_initial_context(context)
        
        # 开始5轮对话
        print("\n5. 开始5轮对话...")
        print("=" * 50)
        
        rounds = conversation_manager.start_conversation(num_rounds=5)
        
        # 显示最终摘要
        print("\n" + "=" * 50)
        print("对话摘要")
        print("=" * 50)
        print(conversation_manager.get_conversation_summary())
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        conversation_manager.save_conversation_to_file(filename)
        
        print(f"\n对话成功完成！")
        print(f"总轮数: {conversation_manager.get_round_count()}")
        print(f"结果已保存到: {filename}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理模型
        print("\n6. 正在清理资源...")
        try:
            if 'llama_model' in locals():
                llama_model.unload_model()
            if 'qwen_model' in locals():
                qwen_model.unload_model()
            print("模型已成功卸载。")
        except Exception as e:
            print(f"清理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
