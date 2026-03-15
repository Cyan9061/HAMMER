"""
Minimal dataset loader.
This module is responsible only for data loading and contains no config logic.
"""

import json
import typing as T
from pathlib import Path
from llama_index.core import Document

from hammer.storage import QAPair
from hammer.logger import logger

class SimpleDataset:
    """Pure dataset interface with no dependency on study configuration."""
    
    def __init__(self, corpus_file: str, qa_file: str, dataset_name: str):
        self.corpus_file = Path(corpus_file)
        self.qa_file = Path(qa_file) 
        self.dataset_name = dataset_name
        
        # Validate file existence.
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file does not exist: {corpus_file}")
        if not self.qa_file.exists():
            raise FileNotFoundError(f"QA file does not exist: {qa_file}")
            
        logger.info("Initialized dataset: %s", dataset_name)
        logger.info("  Corpus: %s", corpus_file)
        logger.info("  QA file: %s", qa_file)
    
    def load_qa_pairs(self) -> T.List[QAPair]:
        """Load QA pairs from JSON or JSONL and resolve `text_ground_truth` ids."""
        qa_pairs = []
        
        # Load the corpus first so ids can be resolved to text.
        corpus_mapping = self._load_corpus_mapping()
        logger.info("Loaded corpus mapping with %s documents", len(corpus_mapping))
        
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Detect the QA file format.
        if content.startswith('[') and content.endswith(']'):
            # Standard JSON array format.
            logger.info("Detected standard JSON array format")
            qa_data = json.loads(content)
        else:
            # JSONL format with one JSON object per line.
            logger.info("Detected JSONL format; parsing line by line")
            qa_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON on line %s: %s", line_num, e)
                        continue
                        
        logger.info("Loaded %s QA entries", len(qa_data))
        
        # Convert records into QAPair objects.
        for i, item in enumerate(qa_data):
            # Unified field mapping.
            question = item.get('query', item.get('question', ''))
            answer = item.get('answer_ground_truth', item.get('answer', ''))
            
            # Resolve supporting_facts from metadata or direct fields.
            supporting_facts = []
            if 'metadata' in item and 'supporting_facts' in item['metadata']:
                supporting_facts = item['metadata']['supporting_facts']
            else:
                supporting_facts = item.get('supporting_facts', [])
            
            # Format the context field.
            context_formatted = []
            if isinstance(supporting_facts, list) and supporting_facts:
                for fact in supporting_facts:
                    if isinstance(fact, list) and len(fact) >= 2:
                        context_formatted.append({"entity": fact[0]})
                    elif isinstance(fact, str):
                        context_formatted.append({"entity": fact})
            
            # Resolve difficulty and question type.
            difficulty = 'unknown'
            qtype = 'multihop'
            if 'metadata' in item:
                metadata = item['metadata']
                difficulty = metadata.get('difficulty', 'unknown')
                qtype = metadata.get('type', 'multihop')
            
            # Resolve text_ground_truth ids to actual corpus text.
            text_ground_truth_content = []
            text_ground_truth_ids = item.get('text_ground_truth', [])
            
            if text_ground_truth_ids and corpus_mapping:
                for doc_id in text_ground_truth_ids:
                    if str(doc_id) in corpus_mapping:
                        text_ground_truth_content.append(corpus_mapping[str(doc_id)])
                    else:
                        logger.warning("Document id %s was not found in the corpus mapping", doc_id)
            
            # Debug output for the first few samples.
            if i < 3:
                logger.info("Sample %s: text_ground_truth_ids=%s", i, text_ground_truth_ids)
                logger.info("Sample %s: resolved text_ground_truth count=%s", i, len(text_ground_truth_content))
            
            # Build the QAPair object.
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                id=item.get('id', f"{self.dataset_name}_{i}"),
                context=context_formatted,
                supporting_facts=supporting_facts,
                difficulty=difficulty,
                qtype=qtype,
                dataset_name=self.dataset_name,
                text_ground_truth=text_ground_truth_content
            ))
                
        return qa_pairs
    
    def load_corpus(self) -> T.List[Document]:
        """Load corpus documents from JSON or JSONL."""
        documents = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Detect the corpus file format.
        if content.startswith('[') and content.endswith(']'):
            # Standard JSON array format.
            logger.info("Detected standard JSON array format for the corpus")
            corpus_data = json.loads(content)
        elif content.startswith('{') and content.count('\n') > 0:
            # JSONL format.
            logger.info("Detected JSONL corpus format; parsing line by line")
            corpus_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        corpus_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse corpus JSON on line %s: %s", line_num, e)
                        continue
        else:
            # Try dictionary format as a fallback.
            try:
                corpus_data = json.loads(content)
                logger.info("Detected dictionary corpus format")
            except json.JSONDecodeError:
                logger.error("Could not recognize corpus file format")
                return []
                        
        logger.info("Loaded %s corpus entries", len(corpus_data))
        
        # Handle the supported corpus layouts.
        if isinstance(corpus_data, dict):
            # Format 1: {doc_id: text, ...}
            for doc_id, text in corpus_data.items():
                documents.append(Document(
                    text=text,
                    metadata={"id": doc_id, "dataset": self.dataset_name}
                ))
        elif isinstance(corpus_data, list):
            # Format 2: [{"id": "...", "text": "..."}, ...] or other list-based layouts.
            for i, doc in enumerate(corpus_data):
                if isinstance(doc, dict):
                    # Unified dict-based handling.
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
                    # Simple string format.
                    documents.append(Document(
                        text=str(doc),
                        metadata={"id": str(i), "dataset": self.dataset_name}
                    ))
        
        return documents
    
    def _load_corpus_mapping(self) -> T.Dict[str, str]:
        """Load the corpus and build an id-to-text mapping."""
        corpus_mapping = {}
        
        if not self.corpus_file.exists():
            logger.warning("Corpus file does not exist: %s", self.corpus_file)
            return corpus_mapping
            
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Detect the corpus file format.
            if content.startswith('[') and content.endswith(']'):
                # Standard JSON array format.
                corpus_data = json.loads(content)
            elif content.startswith('{') and content.count('\n') > 0:
                # JSONL format.
                corpus_data = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            corpus_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                # Try dictionary format.
                corpus_data = json.loads(content)
            
            # Build the id-to-text mapping.
            if isinstance(corpus_data, list):
                for doc in corpus_data:
                    if isinstance(doc, dict) and 'id' in doc and 'text' in doc:
                        corpus_mapping[str(doc['id'])] = doc['text']
            elif isinstance(corpus_data, dict):
                # Dictionary format: {id: text, ...}
                for doc_id, text in corpus_data.items():
                    corpus_mapping[str(doc_id)] = text
                    
        except Exception as e:
            logger.error("Failed to load corpus mapping: %s", e)
            
        return corpus_mapping
    
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        """Compatibility shim for `StudyConfig.dataset`: yield corpus documents."""
        documents = self.load_corpus()
        for doc in documents:
            yield doc
    
    def model_dump(self) -> T.Dict[str, T.Any]:
        """Compatibility shim for the Pydantic-style model_dump interface."""
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
    """Create a SimpleDataset."""
    return SimpleDataset(corpus_file, qa_file, dataset_name)
