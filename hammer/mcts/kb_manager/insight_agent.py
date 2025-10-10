"""
hammer/mcts/kb_manager/insight_agent.py
修复后的insight_agent.py完整版

主要功能：
1. 真实评估结束后根据config和代表性query生成insight（一一对应）
2. 提供简单接口给GPT模拟评估使用
3. 英文优化的prompt系统
4. 修复参数传递问题
"""

import os
import json
import time
import hashlib
import random
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import openai
from hammer.logger import logger
from .graph_memory import QAExecutionNode, ConfigNode, InsightNode

class RAGInsightPrompts:
    """英文优化的RAG洞察提取Prompt系统"""
    
    # === Insight提取Prompt ===
    
    EXTRACT_CONFIG_INSIGHT_SYSTEM = """You are an expert RAG system analyst specialized in extracting actionable insights from configuration evaluations.

Your task is to analyze a specific RAG configuration along with its representative query executions, then generate a comprehensive insight that explains the configuration's performance patterns and provides actionable recommendations.

Focus on:
- Configuration component interactions and their effectiveness
- Query type patterns and performance variations  
- Retrieval and synthesis quality factors
- Actionable recommendations for similar configurations
- Clear causal relationships between configuration choices and outcomes"""

    EXTRACT_CONFIG_INSIGHT_USER = """
## Task: Generate Comprehensive RAG Configuration Insight

### Configuration Being Analyzed:
```json
{config_params}
```

### Performance Summary:
- Average F1 Score: {avg_f1:.3f}
- Average Retrieval Precision: {avg_retrieval_precision:.3f}
- Average Retrieval Recall: {avg_retrieval_recall:.3f}
- Total Evaluations: {total_evaluations}

### Representative Query Executions:

**High-Performing Queries (Top 5):**
{high_performing_queries}

**Low-Performing Queries (Bottom 5):**
{low_performing_queries}

**Medium-Performing Queries (Middle 5):**
{medium_performing_queries}

## Analysis Requirements:

Analyze the configuration and query execution patterns to generate ONE comprehensive insight that:

1. **Identifies Key Success/Failure Factors**: What specific configuration choices led to high vs low performance?

2. **Explains Component Interactions**: How do different RAG components (query decomposition, retrieval method, reranking, etc.) work together in this configuration?

3. **Provides Actionable Recommendations**: What specific changes would improve performance for similar configurations?

4. **Highlights Query Type Patterns**: Are there specific types of questions this configuration handles well or poorly?

## Output Format:

Please provide your analysis in the following structure:

**Title**: [A concise title describing the main insight]

**Performance Analysis**: 
[2-3 sentences explaining the overall performance pattern]

**Key Findings**:
- [Finding 1 about configuration effectiveness]
- [Finding 2 about component interactions] 
- [Finding 3 about query type patterns]

**Actionable Recommendations**:
- [Specific recommendation 1]
- [Specific recommendation 2]
- [Specific recommendation 3]

**Confidence Level**: [High/Medium/Low based on evidence strength]

Your insight:
"""

    # === GPT评估中使用Insight的Prompt ===
    
    EVALUATE_CONFIG_SYSTEM = """You are a professional RAG system performance predictor with deep expertise in retrieval-augmented generation.

Given a RAG configuration, a query, and historical insights from similar configurations, you can accurately predict the expected F1 score based on configuration patterns and domain knowledge."""

    EVALUATE_CONFIG_USER = """
## Task: Predict F1 Score for RAG Configuration

### Query to Process:
"{query}"

### RAG Configuration to Evaluate:
```json
{config_json}
```

### Historical Similar Configurations with Insights:
{similar_configs_with_insights}

### Representative Query Executions from Similar Configs:
{representative_queries}

## Evaluation Framework:

Consider these factors in your prediction:

1. **Configuration-Query Alignment**: How well does the configuration match the query characteristics?
2. **Historical Insight Application**: What do the insights from similar configurations suggest?
3. **Component Synergy**: How effectively do the RAG components work together?
4. **Query Type Patterns**: How does this query type typically perform with similar configurations?

## Prediction Requirements:

- Provide a single F1 score prediction between 0.0 and 1.0
- Consider the multi-hop reasoning requirements of the dataset
- Factor in the insights from similar configurations
- Account for potential failure modes based on historical patterns

## Your Analysis and Prediction:

**Configuration Strengths:**
[Analyze what makes this configuration likely to succeed]

**Potential Weaknesses:**  
[Identify possible failure points]

**Insight Application:**
[Reference how historical insights apply to this scenario]

**Final Prediction:**
Please provide your evaluation as a single number between 0.0 and 1.0.

Your answer must follow this exact format on a new line:
#Predict_Score: [your numerical score]#
"""

    EVALUATE_CONFIG_USER_HYBRID = """
## Task: Refine RAG F1 Score for Test Set Performance Prediction

### Query to Process:
"{query}"

### RAG Configuration to Evaluate:
```json
{config_json}
F1 Score of evaluation of this configuration in tiny train data:true_score={true_score}
```

### Historical Similar Configurations with Insights:
{similar_configs_with_insights}

### Representative Query Executions from Similar Configs:
{representative_queries}

## Evaluation Framework:

Consider these factors in your prediction:

1. **Configuration-Query Alignment**: How well does the configuration match the query characteristics?
2. **Historical Insight Application**: What do the insights from similar configurations suggest?
3. **Component Synergy**: How effectively do the RAG components work together?
4. **Query Type Patterns**: How does this query type typically perform with similar configurations?

## Prediction Requirements:

- Your primary goal is to **adjust the provided `true_score`** to predict the F1 score on a larger, unseen test set.
- The final predicted score **must be within a range of ±0.05 of the given `true_score`**. For example, if the true_score is 0.4537, your prediction must be between 0.4037 and 0.5037.
- The score must be precise to **four decimal places**.
- Factor in the insights from similar configurations to justify your adjustment.
- Account for potential failure modes based on historical patterns.

## Your Analysis and Prediction:

**Configuration Strengths:**
[Analyze what makes this configuration likely to succeed]

**Potential Weaknesses:**  
[Identify possible failure points]

**Insight Application:**
[Reference how historical insights apply to this scenario]

**Final Prediction:**
Based on your analysis, provide the adjusted final score. **Remember, it must be within ±0.05 of the true_score and formatted to four decimal places.**

Your answer must follow this exact format on a new line:
#Predict_Score: [your numerical score]#
"""

class RAGConfigurationAnalyzer:
    """RAG配置分析器 - 简化版"""
    
    @staticmethod
    def extract_config_summary(config_params: Dict[str, Any]) -> str:
        """提取配置摘要"""
        components = []
        
        # 核心组件
        if config_params.get('query_decomposition_enabled'):
            components.append(f"QueryDecomp({config_params.get('query_decomposition_num_queries', 'N/A')})")
        if config_params.get('hyde_enabled'):
            components.append("HyDE")
        
        retrieval = config_params.get('retrieval_method', 'unknown')
        top_k = config_params.get('retrieval_top_k', 'N/A')
        components.append(f"Retrieval({retrieval}-{top_k})")
        
        if config_params.get('reranker_enabled'):
            rerank_k = config_params.get('reranker_top_k', 'N/A')
            components.append(f"Rerank(top-{rerank_k})")
        
        template = config_params.get('template_name', 'unknown')
        components.append(f"Template({template})")
        
        return " → ".join(components)
    
    @staticmethod
    def format_query_execution(qa_execution: QAExecutionNode) -> str:
        """格式化查询执行信息"""
        return f"""
**Question**: "{qa_execution.question}"
**Ground Truth**: "{qa_execution.ground_truth_answer}"
**Predicted**: "{qa_execution.predicted_answer}"
**F1 Score**: {qa_execution.f1_score:.3f}
**Retrieval**: Precision={qa_execution.retrieval_precision:.3f}, Recall={qa_execution.retrieval_recall:.3f}
**RAG Pipeline**: {qa_execution.extract_execution_pattern()}
"""

class InsightAgent:
    """简化的洞察智能体 - 专注于config-insight一一对应"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, 
                 model_name: str = "Qwen/Qwen2.5-72B-Instruct-128K"):
        # 使用提供的凭据或默认值
        api_key = os.getenv("OPENAI_KB_API_KEY", "")#
        api_base = "https://api.siliconflow.cn/v1"#
        
        self.llm_client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = "Qwen/Qwen2.5-72B-Instruct-128K"#model_name
        self.prompts = RAGInsightPrompts()
        self.analyzer = RAGConfigurationAnalyzer()
        
        # 初始化tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize tokenizer in insight agent: {e}")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Token计数"""
        if not self.tokenizer:
            return len(text) // 4
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"⚠️ Token counting error in insight agent: {e}")
            return len(text) // 4
    
    def extract_insights_from_evaluation(self, config_node: ConfigNode, 
                                       qa_executions: List[QAExecutionNode],
                                       existing_insights: List[InsightNode]) -> List[InsightNode]:
        """
        🔥 核心功能：根据config和代表性query生成insight（一一对应）
        """
        logger.info(f"🧠 Starting insight extraction for config {config_node.config_id}")
        logger.info(f"📊 Config performance: F1={config_node.avg_f1_score:.3f}, {len(qa_executions)} QA executions")
        
        if len(qa_executions) == 0:
            logger.warning("⚠️ No QA executions provided, cannot generate insight")
            return []
        
        # 🔥 步骤1：选择代表性query（5高+5低+5中）
        representative_queries = self._select_representative_queries(qa_executions)
        logger.info(f"✅ Selected {len(representative_queries['high'])} high, {len(representative_queries['low'])} low, {len(representative_queries['medium'])} medium queries")
        
        # 🔥 步骤2：使用GPT生成config专属insight
        try:
            insight = self._generate_config_insight_with_gpt(config_node, representative_queries)
            if insight:
                # 确保insight直接关联到当前config
                insight.supporting_config_ids = [config_node.config_id]
                insight.supporting_qa_ids = [qa.qa_id for qa in qa_executions]
                
                logger.info(f"✅ Generated insight for config {config_node.config_id}:")
                logger.info(f"    Title: {insight.title}")
                logger.info(f"    Type: {insight.insight_type}")
                logger.info(f"    Confidence: {insight.confidence_score:.2f}")
                
                return [insight]
            else:
                logger.warning("⚠️ Failed to generate insight with GPT")
                return []
                
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {e}")
            # 创建基于规则的备用insight
            fallback_insight = self._create_fallback_insight(config_node, qa_executions)
            if fallback_insight:
                return [fallback_insight]
            return []
    
    def _select_representative_queries(self, qa_executions: List[QAExecutionNode]) -> Dict[str, List[QAExecutionNode]]:
        """选择代表性查询：5高+5低+5中"""
        if not qa_executions:
            return {'high': [], 'low': [], 'medium': []}
        
        # 按F1分数排序
        sorted_qas = sorted(qa_executions, key=lambda x: x.f1_score, reverse=True)
        total_count = len(sorted_qas)
        
        # 分组选择
        high_queries = sorted_qas[:min(5, total_count)]  # 前5个（或全部）
        low_queries = sorted_qas[-min(5, total_count):] if total_count > 5 else []  # 后5个
        
        # 中等分数的query
        if total_count > 10:
            # 如果总数超过10，从中间部分选择5个
            middle_start = total_count // 3
            middle_end = 2 * total_count // 3
            middle_queries = sorted_qas[middle_start:middle_end]
            medium_queries = middle_queries[:5]
        elif total_count > 5:
            # 如果总数在6-10之间，从中间选择剩余的
            middle_start = min(5, total_count // 2)
            medium_queries = sorted_qas[middle_start:-min(5, total_count - middle_start)]
        else:
            medium_queries = []
        
        logger.info(f"📊 Query selection: Total={total_count}, High={len(high_queries)}, Low={len(low_queries)}, Medium={len(medium_queries)}")
        
        return {
            'high': high_queries,
            'low': low_queries, 
            'medium': medium_queries
        }
    
    def _generate_config_insight_with_gpt(self, config_node: ConfigNode, 
                                        representative_queries: Dict[str, List[QAExecutionNode]]) -> Optional[InsightNode]:
        """使用GPT生成config专属insight"""
        
        # 构建prompt
        high_queries_text = self._format_queries_for_prompt(representative_queries['high'])
        low_queries_text = self._format_queries_for_prompt(representative_queries['low'])
        medium_queries_text = self._format_queries_for_prompt(representative_queries['medium'])
        
        user_prompt = self.prompts.EXTRACT_CONFIG_INSIGHT_USER.format(
            config_params=json.dumps(config_node.config_params, indent=2),
            avg_f1=config_node.avg_f1_score,
            avg_retrieval_precision=config_node.avg_retrieval_precision,
            avg_retrieval_recall=config_node.avg_retrieval_recall,
            total_evaluations=config_node.total_evaluations,
            high_performing_queries=high_queries_text,
            low_performing_queries=low_queries_text,
            medium_performing_queries=medium_queries_text
        )
        
        system_prompt = self.prompts.EXTRACT_CONFIG_INSIGHT_SYSTEM
        
        # 计算token数
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        try:
            logger.info(f"🤖 ===== 开始GPT Insight生成调用 =====")
            logger.info(f"📊 输入统计:")
            logger.info(f"   System prompt token估计: {system_tokens}")
            logger.info(f"   User prompt token估计: {user_tokens}")  
            logger.info(f"   总token估计: {total_tokens}")
            logger.info(f"   模型: {self.model_name}")
            logger.info(f"   配置F1分数: {config_node.avg_f1_score:.3f}")
            logger.info(f"   代表性query数量: 高性能{len(representative_queries['high'])}, 低性能{len(representative_queries['low'])}, 中等性能{len(representative_queries['medium'])}")
            
            # 截断过长的prompt用于日志显示
            display_prompt = user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt
            logger.info(f"📝 User prompt预览: {display_prompt}")
            
            logger.info(f"🚀 发送GPT请求...")
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.,
                max_tokens=2000,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            insight_content = response.choices[0].message.content.strip()
            
            # 详细的响应日志
            logger.info(f"✅ GPT响应成功:")
            logger.info(f"   响应时间: {response_time:.2f}秒")
            logger.info(f"   响应长度: {len(insight_content)} 字符")
            
            # 🔥 记录token使用到全局统计
            from hammer.utils.simple_token_tracker import record_openai_response
            record_openai_response(response, self.model_name)
            
            # token使用统计
            if hasattr(response, 'usage'):
                usage = response.usage
                logger.info(f"   Token使用统计:")
                logger.info(f"     输入token: {getattr(usage, 'prompt_tokens', 'N/A')}")
                logger.info(f"     输出token: {getattr(usage, 'completion_tokens', 'N/A')}")
                logger.info(f"     总token: {getattr(usage, 'total_tokens', 'N/A')}")
            else:
                logger.info(f"   Token使用统计: 不可用")
            
            # 截断response内容用于日志显示
            display_response = insight_content[:300] + "..." if len(insight_content) > 300 else insight_content
            logger.info(f"📝 GPT响应内容预览: {display_response}")
            logger.info(f"🤖 ===== GPT Insight生成调用完成 =====")
            
            # 解析GPT响应生成InsightNode
            return self._parse_gpt_insight_response(insight_content, config_node.config_id)
            
        except Exception as e:
            logger.error(f"❌ GPT insight generation failed: {e}")
            logger.error(f"❌ 调用参数: model={self.model_name}, timeout=60s, max_tokens=2000")
            logger.error(f"❌ Estimated tokens: {total_tokens}")
            if hasattr(e, 'response'):
                logger.error(f"❌ HTTP response: {getattr(e, 'response', 'N/A')}")
            return None
    
    def _format_queries_for_prompt(self, queries: List[QAExecutionNode]) -> str:
        """格式化查询用于prompt"""
        if not queries:
            return "No queries in this performance range."
        
        formatted = ""
        for i, qa in enumerate(queries, 1):
            formatted += f"""
Query {i}:
{self.analyzer.format_query_execution(qa)}
"""
        
        return formatted
    
    def _parse_gpt_insight_response(self, insight_content: str, config_id: str) -> Optional[InsightNode]:
        """解析GPT响应生成InsightNode"""
        try:
            # 提取标题
            title = "RAG Configuration Analysis"
            lines = insight_content.strip().split('\n')
            for line in lines[:5]:  # 在前5行中查找标题
                if line.strip().startswith("**Title"):
                    title_match = line.split(":")
                    if len(title_match) > 1:
                        title = title_match[1].strip().strip("*").strip()
                        break
                elif "title" in line.lower() and len(line.strip()) < 100:
                    title = line.strip()
                    break
            
            # 提取置信度
            confidence_score = 0.8  # 默认置信度
            if "High" in insight_content:
                confidence_score = 0.9
            elif "Low" in insight_content:
                confidence_score = 0.6
            elif "Medium" in insight_content:
                confidence_score = 0.75
            
            # 提取推荐
            recommendation = self._extract_recommendations(insight_content)
            
            # 生成insight ID
            insight_id = self._generate_insight_id(insight_content, config_id)
            
            insight = InsightNode(
                insight_id=insight_id,
                insight_type="config_analysis",
                title=title,
                description=insight_content,
                confidence_score=confidence_score,
                recommendation=recommendation,
                discovery_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                validation_count=1
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"❌ Failed to parse GPT insight response: {e}")
            return None
    
    def _extract_recommendations(self, insight_content: str) -> str:
        """从insight内容中提取推荐"""
        lines = insight_content.split('\n')
        recommendations = []
        
        in_recommendations_section = False
        for line in lines:
            line = line.strip()
            if "recommendation" in line.lower() and ":" in line:
                in_recommendations_section = True
                continue
            elif in_recommendations_section and line.startswith('-'):
                recommendations.append(line[1:].strip())
            elif in_recommendations_section and line and not line.startswith('-'):
                break
        
        if recommendations:
            return "; ".join(recommendations[:3])  # 最多3个推荐
        
        # 备用方法：查找包含推荐词汇的句子
        action_keywords = ['recommend', 'suggest', 'should', 'improve', 'optimize', 'consider']
        for line in lines:
            if any(keyword in line.lower() for keyword in action_keywords):
                return line.strip()[:200]  # 截断以控制长度
        
        return "Continue monitoring performance and consider parameter fine-tuning based on query patterns."
    
    def _create_fallback_insight(self, config_node: ConfigNode, qa_executions: List[QAExecutionNode]) -> Optional[InsightNode]:
        """创建基于规则的备用insight"""
        try:
            config_summary = self.analyzer.extract_config_summary(config_node.config_params)
            avg_f1 = config_node.avg_f1_score
            
            # 生成基于性能的描述
            if avg_f1 > 0.7:
                performance_desc = f"High-performing configuration achieving F1={avg_f1:.3f}"
                insight_type = "high_performance"
                recommendation = "Maintain current configuration and consider minor parameter tuning for optimization."
            elif avg_f1 > 0.5:
                performance_desc = f"Moderate-performing configuration with F1={avg_f1:.3f}"
                insight_type = "moderate_performance" 
                recommendation = "Consider enhancing retrieval components or adjusting query processing parameters."
            else:
                performance_desc = f"Low-performing configuration with F1={avg_f1:.3f}"
                insight_type = "low_performance"
                recommendation = "Significant configuration changes needed, particularly in retrieval and synthesis components."
            
            description = f"""
**Configuration Analysis**: {config_summary}

**Performance Summary**: {performance_desc} across {len(qa_executions)} evaluations.

**Key Observations**: 
- Retrieval precision: {config_node.avg_retrieval_precision:.3f}
- Retrieval recall: {config_node.avg_retrieval_recall:.3f}
- Component interaction patterns require further analysis

**Recommendation**: {recommendation}
"""
            
            insight_id = self._generate_insight_id(description, config_node.config_id)
            
            return InsightNode(
                insight_id=insight_id,
                insight_type=insight_type,
                title=f"Configuration Analysis: {config_summary}",
                description=description.strip(),
                confidence_score=0.7,
                recommendation=recommendation,
                supporting_config_ids=[config_node.config_id],
                supporting_qa_ids=[qa.qa_id for qa in qa_executions],
                discovery_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                validation_count=1
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create fallback insight: {e}")
            return None
    
    def _generate_insight_id(self, content: str, config_id: str) -> str:
        """生成insight ID"""
        combined = f"{config_id}_{content}_{time.time()}"
        hash_val = hashlib.md5(combined.encode()).hexdigest()
        return f"insight_{config_id}_{hash_val[:8]}"
    
    def get_config_linked_insights(self, config_ids: List[str], 
                                  all_insights: List[InsightNode]) -> Dict[str, List[InsightNode]]:
        """
        🔥 为GPT评估提供的简单接口：获取config关联的insight
        """
        config_insights_map = {}
        
        for config_id in config_ids:
            linked_insights = []
            for insight in all_insights:
                if config_id in insight.supporting_config_ids:
                    linked_insights.append(insight)
            
            config_insights_map[config_id] = linked_insights
            logger.info(f"📋 Config {config_id}: found {len(linked_insights)} directly linked insights")
        
        return config_insights_map
    
    def format_insights_for_evaluation(self, config_insights_map: Dict[str, List[InsightNode]]) -> str:
        """
        🔥 为GPT评估格式化insight信息
        """
        if not config_insights_map:
            return "No configuration-linked insights available."
        
        formatted_text = ""
        config_count = 0
        
        for config_id, insights in config_insights_map.items():
            if insights:  # 只显示有insight的config
                config_count += 1
                formatted_text += f"\n**Configuration {config_count} Insights (Config ID: {config_id}):**\n"
                
                for i, insight in enumerate(insights[:2], 1):  # 每个config最多2个insight
                    formatted_text += f"**Insight {i}**: {insight.title}\n"
                    formatted_text += f"  Description: {insight.description[:300]}{'...' if len(insight.description) > 300 else ''}\n"
                    formatted_text += f"  Recommendation: {insight.recommendation}\n"
                    formatted_text += f"  Confidence: {insight.confidence_score:.2f}\n\n"
        
        if not formatted_text.strip():
            return "No insights found for the provided configurations."
        
        return formatted_text

    def select_parameter_choice(self, 
                            parameter_name: str,
                            current_params: Dict[str, Any], 
                            available_options: List[Any],
                            historical_insights: List[str],
                            similar_configs: List[Dict[str, Any]]) -> int:
        """
        🔥 新增：为MCTS提供的参数选择接口
        返回选择的选项索引，失败时返回0（默认选项）
        """
        logger.info(f"🤖 ===== 开始GPT参数选择调用 =====")
        logger.info(f"🎯 参数选择: {parameter_name}")
        logger.info(f"📊 输入统计:")
        logger.info(f"   可选项数量: {len(available_options)}")
        logger.info(f"   历史insights数量: {len(historical_insights)}")
        logger.info(f"   相似配置数量: {len(similar_configs)}")
        logger.info(f"   当前参数数量: {len(current_params)}")
        
        try:
            # 🔥 构建选项列表，给每个选项编号
            options_text = ""
            for i, option in enumerate(available_options):
                options_text += f"{i}: {option}\n"
            logger.info(f"📝 可选项列表:\n{options_text}")
            
            # 构造简化的prompt
            insights_text = "\n".join(f"- {insight[:150]}..." for insight in historical_insights[:3])
            logger.info(f"📚 使用的insights:\n{insights_text}")
            
            configs_text = ""
            for i, config in enumerate(similar_configs[:2], 1):
                configs_text += f"- Config {i}: F1={config['f1_score']:.3f}, evaluations={config['evaluations']}\n"
            logger.info(f"🔍 相似配置:\n{configs_text}")
            
            prompt = f"""You are a RAG system expert. Choose the best parameter value for optimal performance.

    Current configuration:
    {json.dumps(current_params, indent=2)}

    Parameter to choose: {parameter_name}
    Available options:
    {options_text}

    Historical insights:
    {insights_text}

    Similar configurations:
    {configs_text}

    Based on the insights and similar configurations, which option number would likely achieve the highest F1 score? 

    Respond with only the option number (0, 1, 2, etc.), no explanation."""
            
            # 截断prompt用于日志显示
            display_prompt = prompt[:400] + "..." if len(prompt) > 400 else prompt
            logger.info(f"📝 GPT prompt预览:\n{display_prompt}")
            
            logger.info(f"🚀 发送GPT参数选择请求...")
            start_time = time.time()
            
            # 🔥 使用现有的LLM客户端
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            chosen_response = response.choices[0].message.content.strip()
            
            # 详细的响应日志
            logger.info(f"✅ GPT参数选择响应成功:")
            logger.info(f"   响应时间: {response_time:.2f}秒")
            logger.info(f"   响应内容: '{chosen_response}'")
            
            # 🔥 记录token使用到全局统计
            from hammer.utils.simple_token_tracker import record_openai_response
            record_openai_response(response, self.model_name)
            
            # token使用统计
            if hasattr(response, 'usage'):
                usage = response.usage
                logger.info(f"   Token使用统计:")
                logger.info(f"     输入token: {getattr(usage, 'prompt_tokens', 'N/A')}")
                logger.info(f"     输出token: {getattr(usage, 'completion_tokens', 'N/A')}")
                logger.info(f"     总token: {getattr(usage, 'total_tokens', 'N/A')}")
            
            # 🔥 解析GPT返回的序号
            try:
                chosen_index = int(chosen_response)
                if 0 <= chosen_index < len(available_options):
                    logger.info(f"🎯 GPT选择结果: 选项{chosen_index} = '{available_options[chosen_index]}'")
                    logger.info(f"🤖 ===== GPT参数选择调用成功完成 =====")
                    return chosen_index
                else:
                    logger.warning(f"⚠️ GPT返回序号超出范围: {chosen_index}, 可选范围: 0-{len(available_options)-1}")
                    logger.info(f"🔄 使用默认选项0: '{available_options[0]}'")
                    return 0  # 默认选项0
            except ValueError:
                logger.warning(f"⚠️ GPT返回无法解析为数字: '{chosen_response}'")
                logger.info(f"🔄 使用默认选项0: '{available_options[0]}'")
                return 0  # 默认选项0
                
        except Exception as e:
            logger.error(f"❌ GPT参数选择失败: {e}")
            logger.error(f"❌ 调用参数: model={self.model_name}, timeout=30s, max_tokens=50")
            logger.error(f"❌ Parameter: {parameter_name}")
            logger.error(f"❌ Available options: {available_options}")
            if hasattr(e, 'response'):
                logger.error(f"❌ HTTP response: {getattr(e, 'response', 'N/A')}")
            
            logger.info(f"🔄 参数选择失败，使用默认选项0: '{available_options[0] if available_options else 'N/A'}'")
            logger.info(f"🤖 ===== GPT参数选择调用失败完成 =====")
            return 0  # 默认选项0

    def get_knowledge_base_context(self, current_params: Dict[str, Any], graph_memory) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        🔥 新增：获取知识库上下文信息
        返回 (historical_insights, similar_configs)
        """
        historical_insights = []
        similar_configs = []
        
        try:
            # 🔥 添加详细的调试信息
            logger.debug(f"🔍 调试知识库加载: graph_memory={type(graph_memory)}")
            
            if graph_memory is None:
                logger.warning("⚠️ graph_memory 为 None")
                return historical_insights, similar_configs
                
            # 检查 insight_layer
            if hasattr(graph_memory, 'insight_layer'):
                logger.debug(f"✅ graph_memory 有 insight_layer 属性")
                if hasattr(graph_memory.insight_layer, 'nodes'):
                    insights = list(graph_memory.insight_layer.nodes.values())
                    logger.info(f"📚 insight_layer.nodes 包含 {len(insights)} 条insights")
                    
                    if len(insights) == 0:
                        logger.warning("⚠️ insight_layer.nodes 为空，尝试重新加载...")
                        # 🔥 尝试强制重新加载
                        try:
                            graph_memory.insight_layer._load_from_disk()
                            insights = list(graph_memory.insight_layer.nodes.values())
                            logger.info(f"🔄 重新加载后 insight_layer.nodes 包含 {len(insights)} 条insights")
                        except Exception as reload_e:
                            logger.error(f"❌ 重新加载insight_layer失败: {reload_e}")
                    
                    if insights:
                        recent_insights = sorted(insights, key=lambda x: getattr(x, 'discovery_timestamp', '1900-01-01'), reverse=True)[:5]
                        historical_insights = [f"{insight.title}: {insight.description[:200]}..." for insight in recent_insights]
                        logger.info(f"📚 成功加载 {len(historical_insights)} 条最新insights")
                    
                else:
                    logger.warning("⚠️ insight_layer 没有 nodes 属性")
            else:
                logger.warning("⚠️ graph_memory 没有 insight_layer 属性")
                
            # 检查 config_layer  
            if hasattr(graph_memory, 'config_layer'):
                logger.debug(f"✅ graph_memory 有 config_layer 属性")
                if hasattr(graph_memory.config_layer, 'nodes'):
                    configs = list(graph_memory.config_layer.nodes.values())
                    logger.info(f"🔍 config_layer.nodes 包含 {len(configs)} 个configs")
                    
                    if len(configs) == 0:
                        logger.warning("⚠️ config_layer.nodes 为空，尝试重新加载...")
                        # 🔥 尝试强制重新加载
                        try:
                            graph_memory.config_layer._load_from_disk()
                            configs = list(graph_memory.config_layer.nodes.values())
                            logger.info(f"🔄 重新加载后 config_layer.nodes 包含 {len(configs)} 个configs")
                        except Exception as reload_e:
                            logger.error(f"❌ 重新加载config_layer失败: {reload_e}")
                    
                    if configs:
                        logger.debug(f"🔍 开始相似度匹配，当前参数关键词: {list(current_params.keys())}")
                        
                        for i, config in enumerate(configs):
                            similarity_score = 0
                            total_compared = 0
                            
                            # 比较关键参数
                            key_params = ['retrieval_method', 'template_name', 'reranker_enabled', 'hyde_enabled']
                            logger.debug(f"🔍 比较Config {i}: {config.config_id}")
                            
                            for key in key_params:
                                if key in current_params and key in config.config_params:
                                    total_compared += 1
                                    if current_params[key] == config.config_params[key]:
                                        similarity_score += 1
                                    logger.debug(f"  参数 {key}: 当前={current_params.get(key)}, 历史={config.config_params.get(key)}, 匹配={current_params[key] == config.config_params[key] if key in current_params and key in config.config_params else False}")
                            
                            if total_compared > 0:
                                similarity_ratio = similarity_score / total_compared
                                logger.debug(f"  相似度: {similarity_score}/{total_compared} = {similarity_ratio:.2f}")
                                
                                if similarity_ratio > 0.5:  # 至少50%相似
                                    similar_configs.append({
                                        'params': config.config_params,
                                        'f1_score': config.avg_f1_score,
                                        'similarity': similarity_ratio,
                                        'evaluations': config.total_evaluations
                                    })
                                    logger.debug(f"  ✅ 添加到相似配置: F1={config.avg_f1_score:.3f}")
                            else:
                                logger.debug(f"  ❌ 无可比较参数")
                        
                        # 按相似度排序，返回前3个
                        similar_configs.sort(key=lambda x: x['similarity'], reverse=True)
                        similar_configs = similar_configs[:3]
                        logger.info(f"🎯 最终找到 {len(similar_configs)} 个相似Config（相似度>0.5）")
                    
                else:
                    logger.warning("⚠️ config_layer 没有 nodes 属性")
            else:
                logger.warning("⚠️ graph_memory 没有 config_layer 属性")
                
        except Exception as e:
            logger.error(f"❌ 获取知识库上下文失败: {e}", exc_info=True)
        
        return historical_insights, similar_configs