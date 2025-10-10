"""
Multi-hop reasoning evaluation for 2WikiMultiHopQA dataset.
Based on the official 2WikiMultiHopQA evaluation script.
Enhanced with answer extraction for RAG system evaluation.
"""

import re
import string
import typing as T
from collections import Counter

from llama_index.core.evaluation import BaseEvaluator
from pydantic import BaseModel

# from hammer.core import QAPair, normalize_text

class MultiHopEvaluationResult(BaseModel):
    """Comprehensive evaluation result for multi-hop reasoning."""
    # Answer evaluation
    answer_em: float = 0.0
    answer_f1: float = 0.0
    
    # Supporting facts evaluation
    supporting_facts_em: float = 0.0
    supporting_facts_f1: float = 0.0
    
    # Evidence evaluation (if available)
    evidence_em: float = 0.0
    evidence_f1: float = 0.0
    
    # Joint evaluation
    joint_em: float = 0.0
    joint_f1: float = 0.0
    
    # 🔥 新增统一评估指标
    lexical_ac: float = 0.0  # Answer Coverage - 答案覆盖度
    lexical_ff: float = 0.0  # Faithfulness - 忠实度
    mrr: float = 0.0  # Mean Reciprocal Rank - 平均倒数排序
    rouge_l: float = 0.0  # ROUGE-L - 基于最长公共子序列的评估指标
    
    # 数据集标识
    dataset_name: str = ""
    metric_group: str = ""  # "multihop" 或 "domain"

def extract_answer_from_prediction(prediction: str) -> str:
    """
    从完整的prediction中提取Answer:后面的真实答案部分。
    
    此版本修复了提取内容过多，包含了 additional_kwargs 的问题。
    支持处理CompletionResponse对象和字符串格式。
    
    输入格式示例:
    (CompletionResponse(text="...Answer: Ailéan mac Ruaidhrí", additional_kwargs={...}))
    
    Args:
        prediction: 完整的prediction字符串或CompletionResponse对象
        
    Returns:
        提取出的答案部分，如果没有找到Answer:则返回一个合理的备选答案或原始字符串。
    """
    if not prediction:
        return ""
    
    # 将prediction转换为字符串（如果是CompletionResponse对象）
    prediction_str = str(prediction)
    
    # 尝试从CompletionResponse对象中提取text字段
    if "CompletionResponse" in prediction_str and "text=" in prediction_str:
        # 提取text字段内容
        text_match = re.search(r'text=["\']([^"\']*)["\']', prediction_str, re.DOTALL)
        if text_match:
            text_content = text_match.group(1)
            # 从text内容中查找Answer:
            answer_patterns = [
                r"Answer:\s*(.*?)(?=\.|$)",      # 标准格式：Answer: xxx
                r"答案:\s*(.*?)(?=\.|$)",       # 中文格式：答案: xxx  
                r"A:\s*(.*?)(?=\.|$)",           # 简化格式：A: xxx
                r"回答:\s*(.*?)(?=\.|$)",      # 中文简化：回答: xxx
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    # 清理答案（移除标点符号等）
                    return _clean_extracted_answer(answer)
            
            # 如果没有找到Answer:模式，返回text内容
            return _clean_extracted_answer(text_content)
    
    # 备选方案：直接在原始字符串中查找Answer:模式
    answer_patterns = [
        r"Answer:\s*(.*)",      # 标准格式：Answer: xxx
        r"答案:\s*(.*)",       # 中文格式：答案: xxx  
        r"A:\s*(.*)",           # 简化格式：A: xxx
        r"回答:\s*(.*)",      # 中文简化：回答: xxx
    ]
    
    for pattern in answer_patterns:
        # 使用 re.DOTALL 标志让 '.' 可以匹配包括换行在内的任意字符
        match = re.search(pattern, prediction_str, re.IGNORECASE | re.DOTALL)
        if match:
            # group(1) 包含了 "Answer:" 之后的所有内容，因为 .* 是贪婪的
            answer = match.group(1).strip()
            
            # --- 新增的清理逻辑 ---
            # 找到答案和元数据之间的分割点
            stop_phrase = ', additional_kwargs='
            stop_index = answer.find(stop_phrase)
            
            # 如果找到了分割点，就从该位置截断字符串
            if stop_index != -1:
                answer = answer[:stop_index]
            
            # 返回清理后的、去掉首尾空格的答案
            return _clean_extracted_answer(answer.strip())
    
    # --- 改进的备选逻辑 ---
    # 如果上面的所有模式都没有匹配到，尝试提取 `text="..."` 的内容作为备选答案
    text_match = re.search(r'text="((?:[^"\\]|\\.)*)"', prediction_str, re.DOTALL)
    if text_match:
        # 返回 text 字段的内容
        return _clean_extracted_answer(text_match.group(1).strip())

    # 如果连 text 字段都找不到，则返回原始字符串（保持兼容性）
    return _clean_extracted_answer(prediction_str.strip())

def _clean_extracted_answer(answer: str) -> str:
    """
    清理提取出的答案，移除可能的噪声和多余内容。
    
    Args:
        answer: 提取出的原始答案字符串
        
    Returns:
        清理后的答案
    """
    if not answer:
        return ""
    
    # 移除常见的结束标记
    end_markers = [
        r"\n\n.*$",          # 移除双换行后的所有内容
        r"\n-+.*$",         # 移除分隔线后的内容
        r"\nExplanation:.*$", # 移除解释部分
        r"\nReasoning:.*$",   # 移除推理部分
        r"\nContext:.*$",     # 移除上下文部分
        r"\nQuestion:.*$",    # 移除问题部分
    ]
    
    for pattern in end_markers:
        answer = re.sub(pattern, "", answer, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除首尾空白和常见的标点符号
    answer = answer.strip()
    answer = answer.rstrip(".")  # 移除结尾的句号（如果只有一个）
    
    # 如果答案被引号包围，移除引号
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    return answer.strip()

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth."""
    # 首先提取真实答案部分
    extracted_answer = extract_answer_from_prediction(prediction)
    
    # 使用与core.normalize_answer相同的函数
    # from hammer.core import normalize_answer
    pred_normalized = normalize_answer(extracted_answer)
    gt_normalized = normalize_answer(ground_truth)
    return float(pred_normalized == gt_normalized)

def f1_score_string(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth strings using HIPPO2 standard."""
    # 导入logger
    from hammer.logger import logger
    
    # 使用try-catch确保所有日志都能输出
    extracted_answer = ""
    
    try:
        # 首先提取真实答案部分
        extracted_answer = extract_answer_from_prediction(prediction)
    except Exception as e:
        logger.error(f"🚨 答案提取异常: {str(e)}")
        extracted_answer = str(prediction) if prediction else ""
    
    try:
        # 使用与core.f1_score相同的normalize_answer函数
        from collections import Counter
        
        pred_tokens = normalize_answer(extracted_answer).split()
        gt_tokens = normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1_result = float(pred_tokens == gt_tokens)
            # 🔥 增强F1=0时的详细日志
            if f1_result == 0.0:
                logger.info(f"🚨 F1=0 - 空token检测:")
                logger.info(f"  📝 原始预测: {prediction[:200]}{'...' if len(prediction) > 200 else ''}")
                logger.info(f"  🎯 提取答案: '{extracted_answer}'")
                logger.info(f"  ✅ 标准答案: '{ground_truth}'")
                logger.info(f"  🔤 预测tokens: {pred_tokens} (长度: {len(pred_tokens)})")
                logger.info(f"  🔤 标准tokens: {gt_tokens} (长度: {len(gt_tokens)})")
                logger.info(f"  ❌ 结果: F1={f1_result}")
            return f1_result
        
        # 使用Counter保留token频率，符合HIPPO2标准
        common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            # 🔥 增强F1=0时的详细日志输出
            logger.info(f"🚨 F1=0 - 无匹配token:")
            logger.info(f"  📝 原始预测: {prediction[:200]}{'...' if len(prediction) > 200 else ''}")
            logger.info(f"  🎯 提取答案: '{extracted_answer}'")
            logger.info(f"  ✅ 标准答案: '{ground_truth}'")
            logger.info(f"  🔤 预测tokens: {pred_tokens}")
            logger.info(f"  🔤 标准tokens: {gt_tokens}")
            logger.info(f"  🎯 匹配token: {dict(common_tokens)} (数量: {num_common})")
            logger.info(f"  ❌ 结果: F1=0.0")
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # 🔥 当F1较低时也输出详细信息 
        if f1 < 0.3:
            logger.info(f"🟡 低F1分数详情:")
            logger.info(f"  🎯 提取答案: '{extracted_answer}'")
            logger.info(f"  ✅ 标准答案: '{ground_truth}'")
            logger.info(f"  🔤 预测tokens: {pred_tokens}")
            logger.info(f"  🔤 标准tokens: {gt_tokens}")
            logger.info(f"  🎯 匹配token: {dict(common_tokens)} (数量: {num_common})")
            logger.info(f"  📊 计算结果: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
        return f1
        
    except Exception as e:
        logger.error(f"🚨 F1计算异常: {str(e)}")
        #logger.info("❌ 计算异常，返回f1=0.0")
        logger.info("="*80)
        return 0.0

def normalize_answer(s: str) -> str:
    """HIPPO2标准的answer normalization"""
    import re
    import string
    
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_lexical_ac(prediction: str, ground_truth: str, text_ground_truth: T.Optional[T.List[str]] = None) -> float:
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())

    if not gt_tokens:
        return 0.0

    return 1.0 if gt_tokens.issubset(pred_tokens) else 0.0
    
def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    计算ROUGE-L分数，基于最长公共子序列(LCS)
    
    ROUGE-L通过计算预测文本和参考文本之间的最长公共子序列来评估文本质量。
    它考虑了序列的结构信息，比简单的n-gram重叠更能反映文本的连贯性。
    
    Args:
        prediction: 模型预测答案
        reference: 参考答案
        
    Returns:
        ROUGE-L F1分数 (0.0 - 1.0)
    """
    def _lcs_length(x, y):
        """
        使用动态规划计算最长公共子序列长度
        
        Args:
            x: 第一个序列（tokenized text）
            y: 第二个序列（tokenized text）
            
        Returns:
            最长公共子序列的长度
        """
        m, n = len(x), len(y)
        # 创建二维DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # 首先提取答案并标准化
    extracted_prediction = extract_answer_from_prediction(prediction)
    
    # 使用现有的normalize_answer函数进行标准化
    pred_normalized = normalize_answer(extracted_prediction)
    ref_normalized = normalize_answer(reference)
    
    # 分词（按空格分割）
    pred_tokens = pred_normalized.split()
    ref_tokens = ref_normalized.split()
    
    # 处理空文本情况
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # 计算LCS长度
    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    
    # 计算ROUGE-L召回率和精确率
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    
    # 计算F1分数
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # 添加调试日志（仅在必要时）
    from hammer.logger import logger
    logger.debug(f"ROUGE-L计算: pred_tokens={len(pred_tokens)}, ref_tokens={len(ref_tokens)}, "
                f"lcs_len={lcs_len}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    
    return f1

def evaluate_supporting_facts(pred_facts: T.List[T.List], gold_facts: T.List[T.List]) -> T.Tuple[float, float]:
    """
    Evaluate supporting facts.
    
    Args:
        pred_facts: List of [title, sentence_id] predictions
        gold_facts: List of [title, sentence_id] ground truth
        
    Returns:
        Tuple of (EM, F1) scores
    """
    if not pred_facts and not gold_facts:
        return 1.0, 1.0
    
    if not pred_facts or not gold_facts:
        return 0.0, 0.0
    
    # Convert to sets of tuples for comparison
    pred_set = set([tuple(fact) for fact in pred_facts])
    gold_set = set([tuple(fact) for fact in gold_facts])
    
    # Exact match
    em = float(pred_set == gold_set)
    
    # F1 score
    if len(pred_set) == 0 and len(gold_set) == 0:
        f1 = 1.0
    elif len(pred_set) == 0 or len(gold_set) == 0:
        f1 = 0.0
    else:
        common = len(pred_set & gold_set)
        precision = common / len(pred_set)
        recall = common / len(gold_set)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return em, f1

def evaluate_evidences(pred_evidences: T.List[T.List[str]], gold_evidences: T.List[T.List[str]]) -> T.Tuple[float, float]:
    """
    Evaluate evidence triplets.
    
    Args:
        pred_evidences: List of [subject, relation, object] predictions
        gold_evidences: List of [subject, relation, object] ground truth
        
    Returns:
        Tuple of (EM, F1) scores
    """
    if not pred_evidences and not gold_evidences:
        return 1.0, 1.0
    
    if not pred_evidences or not gold_evidences:
        return 0.0, 0.0
    
    # Convert to sets of tuples for comparison
    pred_set = set([tuple(evidence) for evidence in pred_evidences])
    gold_set = set([tuple(evidence) for evidence in gold_evidences])
    
    # Exact match
    em = float(pred_set == gold_set)
    
    # F1 score
    if len(pred_set) == 0 and len(gold_set) == 0:
        f1 = 1.0
    elif len(pred_set) == 0 or len(gold_set) == 0:
        f1 = 0.0
    else:
        common = len(pred_set & gold_set)
        precision = common / len(pred_set)
        recall = common / len(gold_set)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return em, f1

class MultiHopQAEvaluator(BaseEvaluator):
    """Evaluator for multi-hop QA that computes comprehensive metrics."""
    
    def __init__(self, corpus_lookup: T.Optional[T.Dict[str, str]] = None):
        super().__init__()
        self.corpus_lookup = corpus_lookup or {}
    
    def _get_prompts(self):
        """Return empty prompts - not used for this evaluator."""
        return []
    
    def _update_prompts(self, prompts):
        """Update prompts - not used for this evaluator."""
        pass
        
    async def aevaluate(self, query: str, response: str, contexts: T.Optional[T.List[str]] = None, 
                       reference: T.Optional[str] = None, qa_pair=None, **kwargs) -> MultiHopEvaluationResult:
        """Async evaluate - delegates to sync evaluate."""
        return self._evaluate(query, response, contexts, reference, qa_pair, **kwargs)
    
    def _evaluate(self, query: str, response: str, contexts: T.Optional[T.List[str]] = None, 
                  reference: T.Optional[str] = None, qa_pair=None, **kwargs) -> MultiHopEvaluationResult:
        """
        🔥 统一评估 - 根据数据集类型计算相应指标
        
        Args:
            query: The question
            response: The predicted answer (可能包含完整的推理过程)
            contexts: Retrieved contexts (used for MRR calculation)
            reference: The ground truth answer
            qa_pair: The original QAPair with supporting facts and evidences
            
        Returns:
            MultiHopEvaluationResult with all metrics
        """
        # 🔥 添加调试日志
        from hammer.logger import logger
        logger.info(f"🔍 MultiHopQAEvaluator._evaluate 被调用:")
        logger.info(f"  - query: {query[:50]}...")
        logger.info(f"  - response: {response[:50]}...")
        logger.info(f"  - contexts: {len(contexts) if contexts else 0} items")
        logger.info(f"  - reference: {reference}")
        
        # 处理不同格式的qa_pair数据
        if reference is None and qa_pair is not None:
            # 支持字典格式和对象格式
            if isinstance(qa_pair, dict):
                reference = qa_pair.get('answer')
            elif hasattr(qa_pair, 'answer'):
                reference = qa_pair.answer
        
        if reference is None:
            # Cannot evaluate without ground truth
            return MultiHopEvaluationResult()
        
        # 🔥 提取数据集信息和支撑文档
        dataset_name = ""
        text_ground_truth = []
        
        if qa_pair:
            if isinstance(qa_pair, dict):
                dataset_name = qa_pair.get('dataset_name', '')
                text_ground_truth = qa_pair.get('text_ground_truth', [])
            elif hasattr(qa_pair, 'dataset_name'):
                dataset_name = qa_pair.dataset_name
                text_ground_truth = getattr(qa_pair, 'text_ground_truth', [])
        
        # 🔥 确定指标组 - 根据数据集类型
        multihop_datasets = {"2wikimultihopqa", "hotpotqa", "musique"}
        metric_group = "multihop" if dataset_name in multihop_datasets else "domain"
        
        # Answer evaluation - 使用现有的核心指标
        answer_em = exact_match_score(response, reference)
        answer_f1 = f1_score_string(response, reference)
        
        # 🔥 ROUGE-L evaluation - 新增ROUGE-L指标
        rouge_l = compute_rouge_l(response, reference)
        logger.info(f"🎯 ROUGE-L计算完成: {rouge_l:.4f}")
        
        # Supporting facts evaluation (if available) - 保持原有逻辑
        supporting_facts_em = 0.0
        supporting_facts_f1 = 0.0
        if qa_pair and contexts:
            supporting_facts = None
            if isinstance(qa_pair, dict):
                supporting_facts = qa_pair.get('supporting_facts')
            elif hasattr(qa_pair, 'supporting_facts'):
                supporting_facts = qa_pair.supporting_facts
            
            if supporting_facts:
                # Note: In a real implementation, you'd need to extract supporting facts from the response
                # For now, we'll skip this as it requires complex reasoning analysis
                pass
        
        # Evidence evaluation (if available) - 保持原有逻辑
        evidence_em = 0.0
        evidence_f1 = 0.0
        if qa_pair and contexts:
            evidences = None
            if isinstance(qa_pair, dict):
                evidences = qa_pair.get('evidences')
            elif hasattr(qa_pair, 'evidences'):
                evidences = qa_pair.evidences
            
            if evidences:
                # Note: Similarly, evidence extraction from response would be needed
                pass
        
        # 🔥 计算统一评估指标（适用于所有数据集）
        lexical_ac = 0.0
        lexical_ff = 0.0
        mrr = 0.0
        
        try:
            # 🔧 修复：优先使用text_ground_truth，如果为空则使用contexts
            context_docs = text_ground_truth if text_ground_truth else contexts
            
            # 🔥 添加详细调试日志
            logger.info(f"🔍 计算lexical指标:")
            logger.info(f"  - text_ground_truth: {len(text_ground_truth) if text_ground_truth else 0} items")
            logger.info(f"  - contexts: {len(contexts) if contexts else 0} items") 
            logger.info(f"  - context_docs: {len(context_docs) if context_docs else 0} items")
            logger.info(f"  - reference: {reference}")
            
            if context_docs and reference:  # 需要有上下文文档和参考答案
                lexical_ac = compute_lexical_ac(response, reference, context_docs)
                lexical_ff = compute_lexical_ff(response, context_docs, self.corpus_lookup)
                
                logger.info(f"🎯 计算结果: lexical_ac={lexical_ac}, lexical_ff={lexical_ff}")
                
                # MRR计算需要retrieved contexts和ground truth contexts
                if contexts and text_ground_truth:
                    mrr = compute_mrr(response, text_ground_truth, contexts, self.corpus_lookup)
                elif contexts:
                    # 如果没有ground truth，使用contexts本身
                    mrr = compute_mrr(response, contexts, contexts, self.corpus_lookup)
                    
                logger.info(f"🎯 MRR结果: mrr={mrr}")
            elif reference:
                # 🔧 修复：即使没有上下文文档，也能计算AC（基于答案对比）
                lexical_ac = compute_lexical_ac(response, reference)
                logger.info(f"🎯 仅基于答案计算: lexical_ac={lexical_ac}")
                # FF和MRR需要上下文文档，保持0.0
            else:
                logger.info("⚠️ 无法计算lexical指标：缺少reference")
            
            # 对于multihop数据集，可以应用不同的权重或处理逻辑
            if metric_group == "multihop":
                # multihop数据集可能需要调整这些指标的解释
                pass  # 保持原值，但在结果中明确标记metric_group
                
        except Exception as e:
            from hammer.logger import logger
            logger.warning(f"⚠️ 统一指标计算失败 (dataset={dataset_name}): {e}")
            logger.error(f"错误详情: {e}", exc_info=True)
            # 保持默认值0.0
        
        # Joint evaluation - 简化版本，主要基于answer指标
        joint_em = answer_em
        joint_f1 = answer_f1
        
        return MultiHopEvaluationResult(
            answer_em=answer_em,
            answer_f1=answer_f1,
            supporting_facts_em=supporting_facts_em,
            supporting_facts_f1=supporting_facts_f1,
            evidence_em=evidence_em,
            evidence_f1=evidence_f1,
            joint_em=joint_em,
            joint_f1=joint_f1,
            # 🔥 新增指标
            lexical_ac=lexical_ac,
            lexical_ff=lexical_ff,
            mrr=mrr,
            rouge_l=rouge_l,  # 🔥 添加ROUGE-L指标
            dataset_name=dataset_name,
            metric_group=metric_group
        )

# class MultiHopRetrievalEvaluator(BaseEvaluator):
#     """Evaluator specifically for multi-hop retrieval quality."""
    
#     def __init__(self):
#         super().__init__()
    
#     def _get_prompts(self):
#         """Return empty prompts - not used for this evaluator."""
#         return []
    
#     def _update_prompts(self, prompts):
#         """Update prompts - not used for this evaluator."""
#         pass
        
#     async def aevaluate(self, query: str, response: str, contexts: T.List[str] = None,
#                        reference: str = None, qa_pair=None, **kwargs) -> T.Dict[str, float]:
#         """Async evaluate - delegates to sync evaluate."""
#         return self._evaluate(query, response, contexts, reference, qa_pair, **kwargs)
    
#     def _evaluate(self, query: str, response: str, contexts: T.List[str] = None,
#                   reference: str = None, qa_pair=None, **kwargs) -> T.Dict[str, float]:
#         """
#         Evaluate retrieval quality for multi-hop questions.
        
#         Returns:
#             Dictionary with retrieval metrics
#         """
#         if not qa_pair or not contexts:
#             return {"retrieval_recall": 0.0, "retrieval_precision": 0.0}
        
#         # Extract gold evidence from qa_pair - 支持字典和对象格式
#         gold_contexts = []
#         context_data = None
#         gold_evidence = None
        
#         if isinstance(qa_pair, dict):
#             context_data = qa_pair.get('context')
#             gold_evidence = qa_pair.get('gold_evidence')
#         elif hasattr(qa_pair, 'context'):
#             context_data = qa_pair.context
#             gold_evidence = getattr(qa_pair, 'gold_evidence', None)
        
#         if context_data:
#             if isinstance(context_data, dict):
#                 gold_contexts.extend(context_data.values())
#             elif isinstance(context_data, list):
#                 for item in context_data:
#                     if isinstance(item, dict):
#                         gold_contexts.extend(item.values())
#                     else:
#                         gold_contexts.append(str(item))
        
#         if gold_evidence:
#             gold_contexts.extend(gold_evidence)
        
#         if not gold_contexts:
#             return {"retrieval_recall": 0.0, "retrieval_precision": 0.0}
        
#         # Compute retrieval metrics using fuzzy matching
#         from rapidfuzz import fuzz
        
#         # For each gold context, find best match in retrieved contexts
#         matched_gold = 0
#         for gold_text in gold_contexts:
#             best_match = max([fuzz.ratio(gold_text, ctx) / 100.0 for ctx in contexts], default=0.0)
#             if best_match > 0.8:  # Threshold for considering a match
#                 matched_gold += 1
        
#         # For each retrieved context, check if it matches any gold context
#         matched_retrieved = 0
#         for ctx in contexts:
#             best_match = max([fuzz.ratio(ctx, gold_text) / 100.0 for gold_text in gold_contexts], default=0.0)
#             if best_match > 0.8:
#                 matched_retrieved += 1
        
#         recall = matched_gold / len(gold_contexts) if gold_contexts else 0.0
#         precision = matched_retrieved / len(contexts) if contexts else 0.0
        
#         return {
#             "retrieval_recall": recall,
#             "retrieval_precision": precision,
#             "retrieval_f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#         }

# 🔥 新增统一评估指标计算函数

    # return recall

# def compute_ac(prediction: str, ground_truth: str) -> float:
#     """
#     正确的 Answer Correctness (AC) 定义：
#     若 ground_truth 的全部 token 都出现在 prediction 中 → AC=1，否则 0
#     """
#     pred_tokens = set(normalize_answer(prediction).split())
#     gt_tokens = set(normalize_answer(ground_truth).split())

#     if not gt_tokens:
#         return 0.0

#     return 1.0 if gt_tokens.issubset(pred_tokens) else 0.0

def compute_lexical_ff(prediction: str, text_ground_truth: T.List[str], corpus_lookup: T.Optional[T.Dict[str, str]] = None) -> float:
    """计算Faithfulness - 预测答案对检索上下文的忠实度
    
    公式: |Tokens(Prediction) ∩ Tokens(Context)| / |Tokens(Prediction)|
    """
    if not text_ground_truth:
        return 0.0
        
    extracted_answer = extract_answer_from_prediction(prediction)
    pred_tokens = set(normalize_answer(extracted_answer).split())
    
    if not pred_tokens:
        return 0.0
    
    # 🔧 修复：处理document ID格式
    actual_texts = []
    for item in text_ground_truth:
        if corpus_lookup and item in corpus_lookup:
            actual_texts.append(corpus_lookup[item])
        else:
            actual_texts.append(item)  # 假设已经是文本
    
    if not actual_texts:
        return 0.0
    
    # 合并所有检索上下文
    context_text = " ".join(actual_texts)
    context_tokens = set(normalize_answer(context_text).split())
    
    # Faithfulness = |pred ∩ context| / |pred|
    faithful_tokens = len(pred_tokens & context_tokens)
    faithfulness = faithful_tokens / len(pred_tokens)
    
    return faithfulness

def compute_mrr(prediction: str, text_ground_truth: T.List[str], retrieved_contexts: T.Optional[T.List[str]] = None, corpus_lookup: T.Optional[T.Dict[str, str]] = None) -> float:
    """计算Mean Reciprocal Rank
    
    公式: 1 / rank (对于第一个正确文档的排名)
    """
    if not retrieved_contexts or not text_ground_truth:
        return 0.0
    
    # 找到第一个正确文档的排名
    for rank, context in enumerate(retrieved_contexts, 1):
        # 方法1：尝试从上下文中提取文档ID
        context_id = extract_document_id_from_context(context)
        
        if context_id and context_id in text_ground_truth:
            return 1.0 / rank
        
        # 方法2：如果无法提取ID，使用文本相似度匹配
        if corpus_lookup:
            for gold_id in text_ground_truth:
                gold_text = corpus_lookup.get(gold_id, "")
                if gold_text and compute_text_similarity(context, gold_text) > 0.8:
                    return 1.0 / rank
        
        # 方法3：简单的文本包含匹配（备选方案）
        for ground_doc in text_ground_truth:
            if ground_doc in context:
                return 1.0 / rank
    
    return 0.0

def extract_document_id_from_context(context: str) -> str:
    """
    从检索上下文中提取文档ID
    """
    # 方法1：正则表达式匹配常见的ID格式
    id_patterns = [
        r"doc_id:\s*(\S+)",
        r"id:\s*(\S+)",
        r"source:\s*(\S+)",
        r"document:\s*(\S+)",
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 方法2：如果ID嵌入在文本开头
    lines = context.split('\n')
    if lines:
        first_line = lines[0].strip()
        # 假设第一行可能包含ID信息
        if '/' in first_line and ('pdf' in first_line or 'txt' in first_line):
            return first_line
    
    return ""

def compute_text_similarity(text1: str, text2: str) -> float:
    """计算两个文档的Jaccard相似度"""
    if not text1 or not text2:
        return 0.0
        
    tokens1 = set(normalize_answer(text1).split())
    tokens2 = set(normalize_answer(text2).split()) 
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0

# 额外的工具函数，用于其他地方可能需要答案提取的场景
def extract_answer_only(prediction: str) -> str:
    """
    快速提取答案的简化版本，用于其他模块调用。
    
    Args:
        prediction: 完整的prediction字符串
        
    Returns:
        提取出的答案部分
    """
    return extract_answer_from_prediction(prediction)

def test_answer_extraction():
    """
    测试答案提取功能的函数。
    """
    test_cases = [
        {
            "input": "\n---------------------\nContext: Paris is the capital of France.\n---------------------\nQuestion: What is the capital of France?\nAnswer: Paris",
            "expected": "Paris"
        },
        {
            "input": "Based on the context, I can answer this question.\n\nAnswer: The capital is Paris.",
            "expected": "The capital is Paris."
        },
        {
            "input": "这是一个复杂的问题。\n\n答案: 巴黎是法国的首都。",
            "expected": "巴黎是法国的首都。"
        },
        {
            "input": "Let me think about this step by step.\n\nA: 42",
            "expected": "42"
        },
        {
            "input": "Simple answer without any markers",
            "expected": "Simple answer without any markers"
        }
    ]
    
    print("🧪 Testing answer extraction function...")
    for i, test_case in enumerate(test_cases, 1):
        result = extract_answer_from_prediction(test_case["input"])
        status = "✅" if result == test_case["expected"] else "❌"
        print(f"{status} Test {i}: {result == test_case['expected']}")
        if result != test_case["expected"]:
            print(f"   Expected: '{test_case['expected']}'")
            print(f"   Got:      '{result}'")
    
    print("🏁 Answer extraction testing completed.")

if __name__ == "__main__":
    # 运行测试
    test_answer_extraction()
    