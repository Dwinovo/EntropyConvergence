#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def find_log_files(log_dir: str) -> List[str]:
    """查找日志目录中的所有 JSON 文件。"""
    files: List[str] = []
    for entry in os.listdir(log_dir):
        if not entry.endswith('.json'):
            continue
        files.append(os.path.join(log_dir, entry))
    return sorted(files)


def parse_topic_from_filename(filename: str) -> str:
    """从文件名中解析 topic，命名规则：<Model>-<Topic>-<Rounds>-rep<k>.json"""
    base = os.path.basename(filename)
    name = base[:-5] if base.endswith('.json') else base
    parts = name.split('-')
    if len(parts) >= 3:
        return parts[-3]
    # Fallback to whole name if unexpected pattern
    return name


def aggregate_entropy(files: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict[int, Tuple[float, int]]]]:
    """聚合熵信息：
    - topic_stats: 每个主题的总体统计（均值与方差所需的和与平方和）
    - topic_round_stats: 每个主题分轮次的熵和值与计数（用于按轮求均值）
    """
    topic_stats: Dict[str, Dict] = defaultdict(
        lambda: {"overall_sum": 0.0, "overall_sq_sum": 0.0, "overall_count": 0, "num_files": 0}
    )
    topic_round_stats: Dict[str, Dict[int, Tuple[float, int]]] = defaultdict(lambda: defaultdict(lambda: (0.0, 0)))

    for path in files:
        try:
            topic_key = parse_topic_from_filename(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rounds = data.get('conversation_rounds') or []
            # 计数：该 topic 的文件数（一次仅加一）
            topic_stats[topic_key]["num_files"] += 1

            for item in rounds:
                round_number = item.get('round_number')
                entropy = item.get('conditional_entropy_bits_per_token')
                if round_number is None or entropy is None:
                    continue
                # 总体统计：和、平方和、计数
                topic_stats[topic_key]["overall_sum"] += float(entropy)
                topic_stats[topic_key]["overall_sq_sum"] += float(entropy) * float(entropy)
                topic_stats[topic_key]["overall_count"] += 1
                # per round
                s, c = topic_round_stats[topic_key][int(round_number)]
                topic_round_stats[topic_key][int(round_number)] = (s + float(entropy), c + 1)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")
            continue

    return topic_stats, topic_round_stats


def write_mean_variance_csv(out_dir: str, topic_stats: Dict[str, Dict]) -> str:
    """输出每主题的平均熵与方差（总体方差，除以 N）。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'topic_mean_variance.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['topic', 'mean_entropy_bits', 'variance_entropy_bits2'])
        for topic in sorted(topic_stats.keys()):
            stats = topic_stats[topic]
            n = stats.get('overall_count', 0)
            if n > 0:
                mean_v = stats['overall_sum'] / n
                var_v = stats['overall_sq_sum'] / n - mean_v * mean_v
            else:
                mean_v = 0.0
                var_v = 0.0
            writer.writerow([topic, f"{mean_v:.6f}", f"{var_v:.6f}"])
    return out_path


 

def write_pivot_csv(out_dir: str,
                    topic_stats: Dict[str, Dict],
                    topic_round_stats: Dict[str, Dict[int, Tuple[float, int]]],
                    max_round: int = 30) -> str:
    """输出 1..max_round 轮的平均熵透视表，并附总体平均熵（便于快速对照）。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'per_topic_pivot_1_{max_round}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['topic'] + [str(r) for r in range(1, max_round + 1)] + ['average']
        writer.writerow(header)
        for topic in sorted(topic_round_stats.keys()):
            round_map = topic_round_stats[topic]
            row: List[str] = [topic]
            for r in range(1, max_round + 1):
                s, c = round_map.get(r, (0.0, 0))
                avg = (s / c) if c > 0 else 0.0
                row.append(f"{avg:.6f}")
            stats = topic_stats.get(topic, {"overall_sum": 0.0, "overall_count": 0})
            overall_avg = (stats['overall_sum'] / stats['overall_count']) if stats['overall_count'] > 0 else 0.0
            row.append(f"{overall_avg:.6f}")
            writer.writerow(row)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate entropy statistics per topic and per round.')
    parser.add_argument('--log-dir', type=str, default='log', help='Directory containing JSON log files')
    parser.add_argument('--out-dir', type=str, default='output', help='Directory to write analysis outputs')
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    out_dir = os.path.abspath(args.out_dir)

    files = find_log_files(log_dir)
    if not files:
        print(f"[ERROR] No JSON files found in {log_dir}")
        return

    topic_stats, topic_round_stats = aggregate_entropy(files)

    # 导出每主题平均熵与方差（总体方差）
    mean_var_csv = write_mean_variance_csv(out_dir, topic_stats)
    print(f"Wrote topic mean/variance to: {mean_var_csv}")

    # 保留你需要的透视表（1..30 + overall average）
    pivot_csv = write_pivot_csv(out_dir, topic_stats, topic_round_stats, max_round=30)
    print(f"Wrote pivot (1..30 + average) to: {pivot_csv}")

    # 简要打印各主题总体平均熵
    print("\n各主题总体平均熵 (bits):")
    for topic in sorted(topic_stats.keys()):
        stats = topic_stats[topic]
        overall_avg = (stats['overall_sum'] / stats['overall_count']) if stats['overall_count'] > 0 else 0.0
        print(f"  - {topic}: {overall_avg:.6f}")


if __name__ == '__main__':
    main()


