#!/usr/bin/env python3
"""
简化RAG配置评估脚本 - 修复版本 (命令行参数驱动)
专注于对特定配置进行快速RAG评估，无需训练/测试划分
"""
import os
import sys
import json
import time
import logging
import argparse # 新增
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- 移除所有硬编码的全局变量 ---
# USER_PARAMS, CSV_FILE, QA_FILE 等已被移除

# 设置工作目录到项目根目录
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)

# 清理Python路径，确保当前目录优先
current_dir = str(project_root)
sys.path = [p for p in sys.path if 'hammer_backup' not in p and 'Youran/Yuyang/hammer' not in p]
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from hammer.logger import logger
from hammer.flows import Flow
from hammer.tuner.main_tuner_mcts import (
    SimpleDatasetConfig, FlowBuilder, MCTSEvaluationManager
)

class FixedRAGEvaluator:
    """修复后的RAG评估器 - 直接评估指定参数配置"""
    
    def __init__(self, dataset_name: str, corpus_file: str, qa_file: str):
        """
        初始化RAG评估器
        """
        self.dataset_name = dataset_name
        self.corpus_file = corpus_file
        self.qa_file = qa_file
        
        logger.info(f"🚀 初始化修复版RAG评估器")
        logger.info(f"📊 数据集: {dataset_name}")
        logger.info(f"📂 语料库: {corpus_file}")
        logger.info(f"📂 QA文件: {qa_file}")
        logger.info(f"🔧 当前工作目录: {os.getcwd()}")
        
        if not Path(corpus_file).exists():
            raise FileNotFoundError(f"语料库文件不存在: {corpus_file}")
        if not Path(qa_file).exists():
            raise FileNotFoundError(f"QA文件不存在: {qa_file}")
        
        self.dataset_config = self._create_dataset_config()
        self.flow_builder = FlowBuilder(self.dataset_config)
        logger.info(f"✅ RAG评估器初始化完成")
    
    def _create_dataset_config(self):
        """创建数据集配置 - 修复版：避免默认路径污染"""
        qa_count = 0
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                qa_count = sum(1 for line in f if line.strip())
            logger.info(f"📊 QA数据量: {qa_count}条")
        except Exception as e:
            logger.error(f"❌ 读取QA文件失败: {e}")
            qa_count = 100
        
        from hammer.tuner.main_tuner_mcts import SimpleDatasetConfig
        from hammer.mcts.mcts_dataset_loader import create_simple_dataset
        
        class FixedDatasetConfig(SimpleDatasetConfig):
            def __init__(self, dataset_name: str, corpus_file: str, qa_file: str, train_size: int):
                self.dataset_name = dataset_name
                self.corpus_file = corpus_file
                self.qa_file = qa_file  
                self.train_size = train_size
                self.dataset = create_simple_dataset(corpus_file, qa_file, dataset_name)
                
                from hammer.tuner.main_tuner_mcts import SimpleSearchSpace, SimpleTimeoutConfig, SimpleOptimizationConfig
                self.search_space = SimpleSearchSpace()
                self.timeouts = SimpleTimeoutConfig()
                self.optimization = SimpleOptimizationConfig()
                self.toy_mode = False
                self.max_workers = 2
                self.name = f"fixed_{dataset_name}"
                self.model_config = {"extra": "forbid", "yaml_file": None}
        
        config = FixedDatasetConfig(
            dataset_name=self.dataset_name,
            corpus_file=self.corpus_file,
            qa_file=self.qa_file,
            train_size=qa_count
        )
        logger.info(f"📊 自定义数据集配置创建完成")
        return config
    
    def evaluate_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """评估指定的参数配置"""
        logger.info(f"🔧 开始评估RAG配置")
        logger.info(f"📋 配置参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
        start_time = time.time()
        try:
            logger.info("🔧 开始构建RAG Flow...")
            flow = self.flow_builder.build_flow(params)
            logger.info(f"✅ RAG Flow构建成功")
            
            logger.info("🔧 开始RAG评估...")
            evaluation_manager = MCTSEvaluationManager(self.dataset_config)
            objective_value, results = evaluation_manager.evaluate_flow(flow)
            logger.info(f"✅ RAG评估完成，目标值: {objective_value:.4f}")
            
            duration = time.time() - start_time
            results["duration"] = duration
            results["objective_value"] = objective_value
            results["qa_results"] = self._extract_qa_results(results)
            self._print_results_summary(params, results)
            
            return {"success": True, "params": params, "results": results}
        except Exception as e:
            logger.error(f"❌ RAG配置评估失败: {e}")
            import traceback
            logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
            return {"success": False, "params": params, "error": str(e), "duration": time.time() - start_time}
    
    # <--- 以下两个方法是核心修改点 --->

    def _extract_qa_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从评估结果中提取QA结果和每个QA的独立指标，用于CSV记录
        """
        qa_results = []
        try:
            qa_logs = results.get('qa_execution_logs', [])
            if not qa_logs:
                logger.warning("⚠️ 评估结果中没有找到qa_execution_logs，将使用Fallback机制")
                return self._get_qa_fallback_data()
            
            logger.info(f"📊 从已合并指标的qa_logs中提取到{len(qa_logs)}条记录")
            
            for i, log in enumerate(qa_logs):
                try:
                    qa_result = {
                        'query_id': log.get('qa_id', f'query_{i+1}'),
                        'query': log.get('question', ''),
                        'answer_ground_truth': log.get('ground_truth', ''),
                        'predicted_answer': log.get('predicted_answer', ''),
                        
                        'answer_f1': log.get('f1_score', 0.0),
                        'answer_em': log.get('exact_match', 0.0),
                        'lexical_ac': log.get('lexical_ac', 0.0),
                        'lexical_ff': log.get('lexical_ff', 0.0),
                        'rouge_l': log.get('rouge_l', 0.0),
                        'retrieval_precision': log.get('retrieval_precision', 0.0),
                        'retrieval_recall': log.get('retrieval_recall', 0.0),
                        'answer_relevance': log.get('answer_relevance', 0.0),
                        'execution_time': log.get('execution_time', 0.0)
                    }
                    qa_results.append(qa_result)
                except Exception as e:
                    logger.warning(f"⚠️ 处理第{i+1}条QA日志失败: {e}")
                    continue
            logger.info(f"✅ 成功提取{len(qa_results)}条包含完整独立指标的QA结果")
        except Exception as e:
            logger.error(f"❌ 提取QA结果失败: {e}")
            return self._get_qa_fallback_data()
        return qa_results
    
    def _get_qa_fallback_data(self) -> List[Dict[str, Any]]:
        """Fallback机制：当没有qa_execution_logs时，读取原始QA数据"""
        # ... (此方法保持不变) ...
        qa_data = []
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            qa_item = {
                                'query_id': data.get('id', f'query_{line_num}'),
                                'query': data.get('query', ''),
                                'answer_ground_truth': data.get('answer_ground_truth', ''),
                                'predicted_answer': '[评估失败，无生成答案]'
                            }
                            qa_data.append(qa_item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ 解析第{line_num}行JSON失败: {e}")
                            continue
            logger.info(f"📊 Fallback模式：成功加载{len(qa_data)}条QA数据")
        except Exception as e:
            logger.error(f"❌ Fallback模式读取QA文件失败: {e}")
        return qa_data

    def _print_results_summary(self, params: Dict[str, Any], results: Dict[str, Any]):
        """打印结果摘要 - 增强版包含所有MCTS指标"""
        # ... (此方法保持不变) ...
        logger.info(f"\n🎉 RAG配置评估完成 - 结果摘要:")
        logger.info(f"⏱️  总耗时: {results.get('duration', 0):.2f}秒")
        logger.info(f"\n📈 核心性能指标:")
        logger.info(f"   目标值:       {results.get('objective_value', 0.0):.4f}")
        logger.info(f"\n🎯 F1分数:")
        logger.info(f"   Answer F1:    {results.get('answer_f1', 0.0):.4f}")
        logger.info(f"   Joint F1:     {results.get('joint_f1', 0.0):.4f}")
        logger.info(f"\n🎯 EM (Exact Match):")
        logger.info(f"   Answer EM:    {results.get('answer_em', 0.0):.4f}")
        logger.info(f"   Joint EM:     {results.get('joint_em', 0.0):.4f}")
        logger.info(f"\n🎯 统一评估指标:")
        logger.info(f"   Lexical AC:   {results.get('train_lexical_ac', 0.0):.4f}")
        logger.info(f"   Lexical FF:   {results.get('train_lexical_ff', 0.0):.4f}")
        logger.info(f"   ROUGE-L:      {results.get('train_rouge_l', 0.0):.4f}")

    def save_results_to_csv(self, params: Dict[str, Any], results: Dict[str, Any], 
                            csv_file: Optional[str] = None) -> str:
        """
        保存评估结果到CSV文件，每行包含独立的QA结果和其对应的完整指标
        """
        import csv
        try:
            csv_file_path = Path(csv_file)
            csv_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            qa_results = results.get("qa_results", [])
            if not qa_results:
                logger.warning("⚠️ 没有QA结果可保存，CSV文件将为空")
                csv_file_path.touch()
                return str(csv_file_path)

            fieldnames = [
                'query_id', 'query', 'answer_ground_truth', 'predicted_answer',
                'answer_f1', 'answer_em', 
                'lexical_ac', 'lexical_ff', 'rouge_l',
                'retrieval_precision', 'retrieval_recall', 'answer_relevance', 'execution_time',
                'template_name', 'rag_method', 'rag_top_k', 'embedding_model',
                'splitter_method', 'splitter_chunk_exp', 'query_decomposition_enabled', 
                'reranker_enabled', 'full_configuration'
            ]
            
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                common_config_data = {
                    'template_name': params.get('template_name', ''),
                    'rag_method': params.get('rag_method', ''),
                    'rag_top_k': params.get('rag_top_k', 0),
                    'embedding_model': params.get('rag_embedding_model', ''),
                    'splitter_method': params.get('splitter_method', ''),
                    'splitter_chunk_exp': params.get('splitter_chunk_exp', 0),
                    'query_decomposition_enabled': params.get('rag_query_decomposition_enabled', False),
                    'reranker_enabled': params.get('reranker_enabled', False),
                    'full_configuration': json.dumps(params, ensure_ascii=False, separators=(',', ':'))
                }
                
                for qa_result in qa_results:
                    csv_data = qa_result.copy()
                    csv_data.update(common_config_data)
                    writer.writerow(csv_data)
            
            logger.info(f"💾 独立评估结果已保存到CSV文件: {csv_file_path}")
            logger.info(f"💾 共保存了{len(qa_results)}条独立的QA评估记录")
            logger.info(f"💾 整体平均指标: Answer F1={results.get('answer_f1', 0.0):.4f}, AC={results.get('train_lexical_ac', 0.0):.4f}, ROUGE-L={results.get('train_rouge_l', 0.0):.4f}")
            
            return str(csv_file_path)
        except Exception as e:
            logger.error(f"💥 保存CSV文件失败: {e}")
            import traceback
            logger.error(f"💥 详细错误: {traceback.format_exc()}")
            return ""

def create_user_config(user_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据传入的字典创建系统内部配置
    """
    import math
    
    config = {
        "rag_mode": "rag",
        "template_name": user_params.get("template_name"),
        "response_synthesizer_llm": user_params.get("response_synthesizer_llm"),
        "rag_embedding_model": user_params.get("embedding_model"),
        "enforce_full_evaluation": True,
        "splitter_method": user_params.get("splitter_method"),
        "splitter_chunk_overlap_frac": user_params.get("splitter_overlap"),
        "rag_method": user_params.get("retrieval_method"),
        "rag_top_k": user_params.get("retrieval_top_k"),
        "hybrid_bm25_weight": user_params.get("hybrid_bm25_weight"),
        "rag_query_decomposition_enabled": True,
        "rag_query_decomposition_num_queries": user_params.get("query_decomposition_num_queries"),
        "rag_query_decomposition_llm_name": user_params.get("query_decomposition_llm"),
        "rag_fusion_mode": user_params.get("fusion_mode"),
        "reranker_enabled": True,
        "reranker_llm_name": user_params.get("reranker_llm"),
        "reranker_top_k": user_params.get("reranker_top_k"),
        "additional_context_enabled": True,
        "additional_context_num_nodes": user_params.get("additional_context_num_nodes"),
        "hyde_enabled": False,
        "few_shot_enabled": False,
    }
    
    chunk_size = user_params.get("splitter_chunk_size", 512)
    if chunk_size > 0:
        try:
            chunk_exp = round(math.log2(chunk_size))
            config["splitter_chunk_exp"] = chunk_exp
        except (ValueError, OverflowError):
            config["splitter_chunk_exp"] = 9
    else:
        config["splitter_chunk_exp"] = 9
    
    return config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="对指定的RAG配置进行评估。")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称 (例如: hotpotqa)。")
    parser.add_argument("--algorithm-name", type=str, required=True, help="算法/配置的名称 (例如: greedy_m)。")
    parser.add_argument("--params-json", type=str, required=True, help="包含RAG参数的JSON字符串。")
    parser.add_argument("--output-csv", type=str, required=True, help="保存结果的CSV文件完整路径。")
    return parser.parse_args()

def main():
    """主函数 - 执行RAG配置评估"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = parse_args()

    # 从命令行参数动态构建文件路径
    DATASET_NAME = args.dataset
    CORPUS_FILE = f"hammer/tuner/test_dataset/{DATASET_NAME}_corpus_unified.json"
    QA_FILE = f"hammer/tuner/test_dataset/{DATASET_NAME}_qa_unified.json"
    CSV_FILE = args.output_csv
    
    logger.info(f"🚀 开始 RAG 配置评估 ({args.algorithm_name} on {DATASET_NAME})")
    
    try:
        evaluator = FixedRAGEvaluator(
            dataset_name=DATASET_NAME,
            corpus_file=CORPUS_FILE,
            qa_file=QA_FILE
        )
        
        # 从JSON字符串加载参数
        user_params_dict = json.loads(args.params_json)
        user_config = create_user_config(user_params_dict)
        
        result = evaluator.evaluate_config(user_config)
        
        if result["success"]:
            logger.info(f"✅ RAG 配置评估成功!")
            results = result["results"]
            csv_file_path = evaluator.save_results_to_csv(user_config, results, csv_file=CSV_FILE)
            if csv_file_path:
                logger.info(f"📊 CSV 文件已保存: {csv_file_path}")
                
            logger.info(f"\n🎯 最终评估结果摘要:")
            logger.info(f"   目标值:       {results.get('objective_value', 0.0):.4f}")
            logger.info(f"   Answer F1:    {results.get('answer_f1', 0.0):.4f}")
            logger.info(f"   Joint F1:     {results.get('joint_f1', 0.0):.4f}")
            logger.info(f"   Answer EM:    {results.get('answer_em', 0.0):.4f}")
            logger.info(f"   Joint EM:     {results.get('joint_em', 0.0):.4f}")
            logger.info(f"   Lexical AC:   {results.get('train_lexical_ac', 0.0):.4f}")
            logger.info(f"   Lexical FF:   {results.get('train_lexical_ff', 0.0):.4f}")
            logger.info(f"   ROUGE-L:      {results.get('train_rouge_l', 0.0):.4f}")
            logger.info(f"   总耗时:       {results.get('duration', 0.0):.2f}秒")
        else:
            logger.error(f"❌ RAG 配置评估失败: {result['error']}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
    
    logger.info(f"\n🎉 RAG 配置评估完成 ({args.algorithm_name} on {DATASET_NAME})!")

if __name__ == "__main__":
    main()