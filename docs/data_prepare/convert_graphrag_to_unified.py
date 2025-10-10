#!/usr/bin/env python3
"""
GraphRAG-Benchmark数据转换为Unified格式
采用方案1：整个医学文档作为单一corpus条目
"""

import json
import os
from typing import Dict, List, Any

class GraphRAGToUnifiedConverter:
    def __init__(self):
        # 问题类型映射
        self.TYPE_MAPPING = {
            "Fact Retrieval": "fact_retrieval",
            "Complex Reasoning": "complex_reasoning", 
            "Contextual Summarization": "contextual_summarization",
            "Contextual Summarize": "contextual_summarization",  # 处理拼写差异
            "Creative Generation": "creative_generation"
        }
        
        # 难度等级映射  
        self.DIFFICULTY_MAPPING = {
            "Fact Retrieval": "easy",
            "Complex Reasoning": "hard",
            "Contextual Summarization": "medium",
            "Contextual Summarize": "medium",  # 处理拼写差异
            "Creative Generation": "hard"
        }
        
        # 统一的corpus ID
        self.CORPUS_ID_MED = "graph_med_unified"
        self.CORPUS_ID_NOVEL = "graph_novel_unified"
        
    def convert_corpus(self, corpus_data: List[Dict], domain: str) -> List[Dict]:
        """
        转换corpus数据
        将所有文档合并为单一corpus条目
        """
        if domain == "medical":
            corpus_id = self.CORPUS_ID_MED
        elif domain == "novel":
            corpus_id = self.CORPUS_ID_NOVEL
        else:
            raise ValueError(f"Unsupported domain: {domain}")
            
        # 合并所有文档内容
        all_text = []
        for item in corpus_data:
            if "context" in item:
                all_text.append(item["context"])
        
        combined_text = "\n\n".join(all_text)
        
        return [{"id": corpus_id, "text": combined_text}]
    
    def convert_qa(self, qa_data: List[Dict], domain: str) -> List[Dict]:
        """
        转换QA数据为unified格式
        """
        if domain == "medical":
            corpus_id = self.CORPUS_ID_MED
        elif domain == "novel": 
            corpus_id = self.CORPUS_ID_NOVEL
        else:
            raise ValueError(f"Unsupported domain: {domain}")
            
        unified_qa = []
        
        for item in qa_data:
            # 基本字段映射
            unified_item = {
                "id": f"graph_{domain}_{item['id']}",
                "query": item["question"],
                "answer_ground_truth": item["answer"],
                "text_ground_truth": [corpus_id],  # 所有QA都指向同一个corpus
                "metadata": {
                    "type": self.TYPE_MAPPING.get(item["question_type"], "unknown"),
                    "difficulty": self.DIFFICULTY_MAPPING.get(item["question_type"], "medium"),
                    "supporting_facts": [[corpus_id, 0]],  # 简化处理，都指向第0段
                    "evidences": [],  # 暂时为空，后续可扩展
                    "entity_ids": "",  # 暂时为空
                    "original_source": "GraphRAG-Benchmark",
                    "original_type": item["question_type"]
                }
            }
            
            unified_qa.append(unified_item)
            
        return unified_qa
    
    def save_as_jsonl(self, data: List[Dict], output_path: str):
        """
        保存为JSONL格式
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"✅ 已保存 {len(data)} 条记录到: {output_path}")
    
    def convert_domain(self, 
                      corpus_file: str, 
                      qa_file: str, 
                      domain: str,
                      output_dir: str):
        """
        转换特定领域的数据
        """
        print(f"\n🔄 开始转换 {domain} 领域数据...")
        
        # 读取corpus数据
        print(f"📖 读取corpus文件: {corpus_file}")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        print(f"📊 Corpus数据: {len(corpus_data)} 条文档")
            
        # 读取QA数据
        print(f"📖 读取QA文件: {qa_file}")
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"📊 QA数据: {len(qa_data)} 条问题")
        
        # 转换数据
        unified_corpus = self.convert_corpus(corpus_data, domain)
        unified_qa = self.convert_qa(qa_data, domain)
        
        # 保存结果
        corpus_output = os.path.join(output_dir, f"graph_{domain}_corpus_unified.json")
        qa_output = os.path.join(output_dir, f"graph_{domain}_qa_unified.json")
        
        self.save_as_jsonl(unified_corpus, corpus_output)
        self.save_as_jsonl(unified_qa, qa_output)
        
        return len(unified_corpus), len(unified_qa)

def main():
    """
    主函数：执行完整的转换流程
    """
    converter = GraphRAGToUnifiedConverter()
    
    # 设置路径
    base_dir = "docs/data_prepare/GraphRAG-Benchmark/Datasets"
    output_dir = "docs/dataset/graph_unified"
    
    print("🚀 GraphRAG-Benchmark到Unified格式转换工具")
    print("=" * 60)
    
    total_corpus = 0
    total_qa = 0
    
    # 转换医学领域数据
    try:
        corpus_count, qa_count = converter.convert_domain(
            corpus_file=os.path.join(base_dir, "Corpus/medical.json"),
            qa_file=os.path.join(base_dir, "Questions/medical_questions.json"),
            domain="medical",
            output_dir=output_dir
        )
        total_corpus += corpus_count
        total_qa += qa_count
    except Exception as e:
        print(f"❌ 医学领域转换失败: {e}")
    
    # 转换文学领域数据
    try:
        corpus_count, qa_count = converter.convert_domain(
            corpus_file=os.path.join(base_dir, "Corpus/novel.json"),
            qa_file=os.path.join(base_dir, "Questions/novel_questions.json"),
            domain="novel",
            output_dir=output_dir
        )
        total_corpus += corpus_count
        total_qa += qa_count
    except Exception as e:
        print(f"❌ 文学领域转换失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 转换完成!")
    print(f"📊 总计: {total_corpus} 个corpus条目, {total_qa} 个QA对")
    print(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    main()