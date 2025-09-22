#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path


def main() -> None:
    project_root: Path = Path(__file__).resolve().parents[1]
    main_py: Path = project_root / "main.py"
    topic_dir: Path = project_root / "topic"
    log_dir: Path = project_root / "log"

    # 固定参数
    model_path: str = "/root/autodl-fs/Qwen2-7B-Instruct"
    rounds: int = 30

    # 目录检查/创建
    if not main_py.exists():
        print(f"错误：找不到 main.py: {main_py}")
        sys.exit(1)
    if not topic_dir.exists():
        print(f"错误：找不到 topic 目录: {topic_dir}")
        sys.exit(1)
    log_dir.mkdir(parents=True, exist_ok=True)

    topic_files = sorted(topic_dir.glob("*.txt"))
    if not topic_files:
        print(f"警告：未在 {topic_dir} 找到任何 .txt 主题文件。")
        return

    for topic_file in topic_files:
        topic_stem: str = topic_file.stem
        print(f"== Topic: {topic_stem} ==")
        for rep_index in (1, 2, 3):
            name_suffix: str = f"rep{rep_index}"
            cmd = [
                sys.executable, str(main_py),
                "--model_path", model_path,
                "--topic", topic_stem,
                "--rounds", str(rounds),
                "--name", name_suffix,
            ]
            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(
                    f"运行失败: topic={topic_stem}, name={name_suffix}, returncode={result.returncode}"
                )
                # 不中断，继续后续实验

    print("All experiments completed.")


if __name__ == "__main__":
    main()


