#!/usr/bin/env python
"""
诊断失败的 LLM 响应
从 debug_llm_responses.txt 中提取所有响应，分析哪些窗口解析失败
"""

import os
import re
import sys
import json
from datetime import datetime

# 调试文件路径
DEBUG_FILE = os.path.join("llog", "debug_llm_responses.txt")
OUTPUT_FILE = os.path.join("llog", "failed_responses_analysis.txt")

def parse_responses_from_file(filepath):
    """从调试文件中解析所有响应记录"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分隔符分割记录
    records = re.split(r'={80}\n', content)
    parsed_records = []

    for record in records:
        if not record.strip():
            continue

        # 提取基本信息
        time_match = re.search(r'时间: (.+?)\n', record)
        model_match = re.search(r'模型: (.+?)\n', record)
        window_match = re.search(r'窗口ID: (\d+)\n', record)
        prompt_len_match = re.search(r'Prompt长度: (\d+) 字符\n', record)
        resp_len_match = re.search(r'响应长度: (\d+) 字符\n', record)

        # 提取原始响应（在 "--- 原始响应 ---" 之后）
        raw_match = re.search(r'--- 原始响应 ---\n(.*?)(?=\n--- |\n={80}|$)', record, re.DOTALL)
        raw_content = raw_match.group(1).strip() if raw_match else ""

        # 提取错误信息（如果有）
        error_match = re.search(r'--- 错误信息 ---\n(.*?)(?=\n={80}|$)', record, re.DOTALL)
        error_msg = error_match.group(1).strip() if error_match else ""

        # 检查是否有 "修复后的 JSON" 部分（说明尝试过修复）
        has_repaired = "--- 修复后的 JSON ---" in record

        # 判断是否解析成功：如果没有错误信息，且原始响应以 { 开头且以 } 结尾
        is_success = False
        if raw_content:
            stripped = raw_content.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    json.loads(stripped)
                    is_success = True
                except:
                    pass

        parsed_records.append({
            'timestamp': time_match.group(1) if time_match else "未知",
            'model': model_match.group(1) if model_match else "未知",
            'window_id': int(window_match.group(1)) if window_match else None,
            'prompt_len': int(prompt_len_match.group(1)) if prompt_len_match else 0,
            'resp_len': int(resp_len_match.group(1)) if resp_len_match else 0,
            'raw_content': raw_content,
            'error_msg': error_msg,
            'has_repaired': has_repaired,
            'is_success': is_success,
            'full_record': record
        })

    return parsed_records


def analyze_failed_responses(records):
    """分析失败的响应"""
    failed = [r for r in records if not r['is_success'] and r['raw_content']]
    success = [r for r in records if r['is_success']]

    print("\n" + "=" * 80)
    print(f"📊 响应分析报告")
    print("=" * 80)
    print(f"总记录数: {len(records)}")
    print(f"成功解析: {len(success)}")
    print(f"失败: {len(failed)}")

    if failed:
        print("\n❌ 失败的窗口:")
        for r in failed:
            print(f"  窗口 {r['window_id']}: {r['timestamp']}")
            if r['error_msg']:
                print(f"    错误: {r['error_msg'][:100]}...")

    return failed


def save_failed_analysis(failed_records, output_file):
    """将失败的响应保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"失败响应分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(failed_records):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"【失败记录 {i+1}】窗口 {r['window_id']}\n")
            f.write(f"时间: {r['timestamp']}\n")
            f.write(f"模型: {r['model']}\n")
            f.write(f"响应长度: {r['resp_len']} 字符\n")
            f.write(f"错误信息: {r['error_msg']}\n")
            f.write(f"尝试修复: {'是' if r['has_repaired'] else '否'}\n")
            f.write("\n--- 完整原始响应 ---\n")
            f.write(r['raw_content'] + "\n")
            f.write("\n--- 响应开头（前500字符）---\n")
            f.write(r['raw_content'][:500] + "...\n")

    print(f"\n📁 失败响应已保存到: {output_file}")


def main():
    print("=" * 80)
    print("🔍 诊断失败的 LLM 响应")
    print("=" * 80)

    if not os.path.exists(DEBUG_FILE):
        print(f"❌ 未找到调试文件: {DEBUG_FILE}")
        print("请先运行训练程序，产生 debug_llm_responses.txt 文件。")
        return

    records = parse_responses_from_file(DEBUG_FILE)
    if not records:
        print("❌ 未能解析任何记录")
        return

    failed = analyze_failed_responses(records)

    if failed:
        save_failed_analysis(failed, OUTPUT_FILE)

        # 打印失败响应的简短预览
        print("\n" + "=" * 80)
        print("📝 失败响应预览（前200字符）")
        print("=" * 80)
        for r in failed[:3]:  # 只显示前3个
            print(f"\n【窗口 {r['window_id']}】")
            print(f"原始响应开头: {r['raw_content'][:200]}...")

        print(f"\n💡 完整内容请查看: {OUTPUT_FILE}")
    else:
        print("\n✅ 没有找到失败的响应！所有响应都已成功解析。")


if __name__ == "__main__":
    main()