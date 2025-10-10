"""
Knowledge Base Manager for MCTS-RAG Optimization

This package implements the three-layer graph memory system:
1. Query Layer: Detailed QA execution records with complete RAG flow
2. Config Layer: Configuration performance analysis and relationships  
3. Insight Layer: Extracted insights and rules for optimization guidance
"""

from .graph_memory import (
    QAExecutionNode,
    ConfigNode, 
    InsightNode,
    QueryLayer,
    ConfigLayer,
    InsightLayer,
    GraphMemoryRAGMCTS
)

from .insight_agent import (
    InsightAgent,
    RAGInsightPrompts
)

from .enhanced_evaluator import (
    EnhancedGPTSimulationEvaluator
)

__all__ = [
    # Core data structures
    'QAExecutionNode',
    'ConfigNode',
    'InsightNode',
    
    # Memory layers
    'QueryLayer',
    'ConfigLayer', 
    'InsightLayer',
    'GraphMemoryRAGMCTS',
    
    # Insight system
    'InsightAgent',
    'RAGInsightPrompts',
    
    # Enhanced evaluator
    'EnhancedGPTSimulationEvaluator'
]

__version__ = "1.0.0"