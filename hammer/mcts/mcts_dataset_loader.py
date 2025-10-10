"""
极简数据集加载器 - 真正解耦版本
只负责数据加载，不包含任何配置逻辑
"""

import json
import typing as T
from pathlib import Path
from llama_index.core import Document

from hammer.storage import QAPair
from hammer.logger import logger

class SimpleDataset:
    """纯粹的数据集接口 - 不依赖任何配置"""
    
    def __init__(self, corpus_file: str, qa_file: str, dataset_name: str):
        self.corpus_file = Path(corpus_file)
        self.qa_file = Path(qa_file) 
        self.dataset_name = dataset_name
        
        # 验证文件存在
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"❌ 语料库文件不存在: {corpus_file}")
        if not self.qa_file.exists():
            raise FileNotFoundError(f"❌ QA文件不存在: {qa_file}")
            
        logger.info(f"✅ 数据集初始化: {dataset_name}")
        logger.info(f"  📂 语料库: {corpus_file}")
        logger.info(f"  📂 问答对: {qa_file}")
    
    def load_qa_pairs(self) -> T.List[QAPair]:
        """加载QA对 - 支持JSON和JSONL格式，处理text_ground_truth字段"""
        qa_pairs = []
        
        # 🔥 首先加载语料库数据，创建ID到文本的映射
        corpus_mapping = self._load_corpus_mapping()
        logger.info(f"📚 语料库映射加载完成，共 {len(corpus_mapping)} 个文档")
        
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 尝试判断文件格式
        if content.startswith('[') and content.endswith(']'):
            # 标准JSON数组格式
            logger.info(f"📋 检测到标准JSON数组格式")
            qa_data = json.loads(content)
        else:
            # JSONL格式（每行一个JSON对象）
            logger.info(f"📋 检测到JSONL格式，逐行解析")
            qa_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                        continue
                        
        logger.info(f"📊 成功加载 {len(qa_data)} 个QA条目")
        
        # 转换为QAPair对象
        for i, item in enumerate(qa_data):
            # 处理统一格式字段映射
            question = item.get('query', item.get('question', ''))  # 支持query或question字段
            answer = item.get('answer_ground_truth', item.get('answer', ''))  # 支持多种答案字段名
            
            # 处理supporting_facts - 来自metadata或直接字段
            supporting_facts = []
            if 'metadata' in item and 'supporting_facts' in item['metadata']:
                supporting_facts = item['metadata']['supporting_facts']
            else:
                supporting_facts = item.get('supporting_facts', [])
            
            # 格式化context
            context_formatted = []
            if isinstance(supporting_facts, list) and supporting_facts:
                for fact in supporting_facts:
                    if isinstance(fact, list) and len(fact) >= 2:
                        context_formatted.append({"entity": fact[0]})
                    elif isinstance(fact, str):
                        context_formatted.append({"entity": fact})
            
            # 处理难度和类型
            difficulty = 'unknown'
            qtype = 'multihop'
            if 'metadata' in item:
                metadata = item['metadata']
                difficulty = metadata.get('difficulty', 'unknown')
                qtype = metadata.get('type', 'multihop')
            
            # 🔥 处理text_ground_truth字段 - 将ID映射到实际文本内容
            text_ground_truth_content = []
            text_ground_truth_ids = item.get('text_ground_truth', [])
            
            if text_ground_truth_ids and corpus_mapping:
                for doc_id in text_ground_truth_ids:
                    if str(doc_id) in corpus_mapping:
                        text_ground_truth_content.append(corpus_mapping[str(doc_id)])
                    else:
                        logger.warning(f"⚠️ 语料库中未找到文档ID: {doc_id}")
            
            # 🔍 调试信息
            if i < 3:  # 只为前3个样本输出调试信息
                logger.info(f"🔍 样本 {i}: text_ground_truth_ids={text_ground_truth_ids}")
                logger.info(f"🔍 样本 {i}: text_ground_truth_content数量={len(text_ground_truth_content)}")
            
            # 创建QAPair对象
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                id=item.get('id', f"{self.dataset_name}_{i}"),
                context=context_formatted,
                supporting_facts=supporting_facts,
                difficulty=difficulty,
                qtype=qtype,
                dataset_name=self.dataset_name,
                text_ground_truth=text_ground_truth_content  # 🔥 添加映射后的文本内容
            ))
                
        return qa_pairs
    
    def load_corpus(self) -> T.List[Document]:
        """加载语料库文档 - 支持JSON和JSONL格式"""
        documents = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 尝试判断文件格式
        if content.startswith('[') and content.endswith(']'):
            # 标准JSON数组格式
            logger.info(f"📋 检测到标准JSON数组格式（语料库）")
            corpus_data = json.loads(content)
        elif content.startswith('{') and content.count('\n') > 0:
            # JSONL格式（每行一个JSON对象）
            logger.info(f"📋 检测到JSONL格式（语料库），逐行解析")
            corpus_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        corpus_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ 语料库第{line_num}行JSON解析失败: {e}")
                        continue
        else:
            # 尝试作为字典格式解析
            try:
                corpus_data = json.loads(content)
                logger.info(f"📋 检测到字典格式（语料库）")
            except json.JSONDecodeError:
                logger.error(f"❌ 无法识别语料库文件格式")
                return []
                        
        logger.info(f"📊 成功加载 {len(corpus_data)} 个语料库条目")
        
        # 处理不同的数据格式
        if isinstance(corpus_data, dict):
            # 格式1: {doc_id: text, ...}
            for doc_id, text in corpus_data.items():
                documents.append(Document(
                    text=text,
                    metadata={"id": doc_id, "dataset": self.dataset_name}
                ))
        elif isinstance(corpus_data, list):
            # 格式2: [{"id": "...", "text": "..."}, ...] 或其他列表格式
            for i, doc in enumerate(corpus_data):
                if isinstance(doc, dict):
                    # 统一格式处理
                    text = doc.get('text', str(doc))
                    doc_id = doc.get('id', doc.get('title', f'doc_{i}'))
                    title = doc.get('title', doc_id)
                    
                    documents.append(Document(
                        text=text,
                        metadata={
                            "title": title, 
                            "id": str(doc_id), 
                            "dataset": self.dataset_name
                        }
                    ))
                else:
                    # 简单字符串格式
                    documents.append(Document(
                        text=str(doc),
                        metadata={"id": str(i), "dataset": self.dataset_name}
                    ))
        
        return documents
    
    def _load_corpus_mapping(self) -> T.Dict[str, str]:
        """加载语料库并创建ID到文本的映射"""
        corpus_mapping = {}
        
        if not self.corpus_file.exists():
            logger.warning(f"⚠️ 语料库文件不存在: {self.corpus_file}")
            return corpus_mapping
            
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # 尝试判断文件格式
            if content.startswith('[') and content.endswith(']'):
                # 标准JSON数组格式
                corpus_data = json.loads(content)
            elif content.startswith('{') and content.count('\n') > 0:
                # JSONL格式
                corpus_data = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            corpus_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                # 尝试作为字典格式解析
                corpus_data = json.loads(content)
            
            # 创建ID到文本的映射
            if isinstance(corpus_data, list):
                for doc in corpus_data:
                    if isinstance(doc, dict) and 'id' in doc and 'text' in doc:
                        corpus_mapping[str(doc['id'])] = doc['text']
            elif isinstance(corpus_data, dict):
                # 如果是字典格式 {id: text, ...}
                for doc_id, text in corpus_data.items():
                    corpus_mapping[str(doc_id)] = text
                    
        except Exception as e:
            logger.error(f"❌ 语料库加载失败: {e}")
            
        return corpus_mapping
    
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        """兼容StudyConfig.dataset接口：返回语料库文档的迭代器"""
        documents = self.load_corpus()
        for doc in documents:
            yield doc
    
    def model_dump(self) -> T.Dict[str, T.Any]:
        """兼容Pydantic模型接口：返回模型字典"""
        return {
            "xname": "simple_dataset",
            "dataset_name": self.dataset_name,
            "corpus_file": str(self.corpus_file),
            "qa_file": str(self.qa_file),
            "partition_map": {"test": "test", "train": "train"},
            "subset": "default",
            "grounding_data_path": str(self.corpus_file)
        }

def create_simple_dataset(corpus_file: str, qa_file: str, dataset_name: str) -> SimpleDataset:
    """创建简单数据集"""
    return SimpleDataset(corpus_file, qa_file, dataset_name)