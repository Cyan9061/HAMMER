import torch
from transformers import pipeline
from typing import List
from hammer.Rerank.BasicRerank import BaseRanking
from hammer.Schema.DocumentSchema import Document
from tqdm import tqdm  # Import tqdm for progress tracking
import copy

class EchoRankReranker(BaseRanking):
    """
    Implements **EchoRank**, a reranking method designed for budget-constrained text reranking using Large Language Models (LLMs).

    EchoRank uses a two-stage approach to balance reranking performance with computational cost:
    
    1. **Binary Classification Stage**: Determines whether each passage is relevant to the query.
    2. **Pairwise Ranking Stage**: Directly compares relevant passages to refine rankings.

    This method is particularly suitable when computational or token budgets are limited.

    Attributes:
        method (str, optional): The reranking method name.
        model_name (str): The Hugging Face pretrained model name (default: `"google/flan-t5-large"`).
        type (str): Type of budget constraint applied (default: `"cheap"`).
        budget_tokens (int): Total token budget allocated for reranking (default: `4000`).
        budget_split_x (float): Fraction of token budget for the binary classification stage (default: `0.5`).
        budget_split_y (float): Fraction of token budget for the pairwise ranking stage (default: `0.5`).
        total_passages (int): Maximum number of passages processed per query (default: `50`).
        device (str): Computational device (`"cuda"` or `"cpu"`).
        model (transformers.Pipeline): Hugging Face pipeline for text-to-text generation.

    References:
        - **Rashid et al.** *EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models.*  
          [Paper](https://arxiv.org/abs/2402.10866)
        - [Original Implementation](https://github.com/shihabrashid-ucr/EcoRank/tree/main)

    See Also:
        - `Reranking`: Main interface for reranking models, including `EchoRankReranker`.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.reranking import Reranking

        question = Question("What is climate change?")
        answers = Answer(["Climate change refers to long-term shifts in temperatures and weather patterns."])
        contexts = [
            Context(text="Climate change is mainly caused by human activities.", id=1),
            Context(text="Deforestation contributes to global warming.", id=2),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        model = Reranking(method='echorank', model_name='flan-t5-large')
        model.rank([document])

        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def __init__(self, method=None, model_name=None, **kwargs):
        """
        Initializes the **EchoRankReranker** instance.

        Args:
            method (str, optional): The reranking method name.
            model_name (str, optional): Hugging Face pretrained model (default: `"google/flan-t5-large"`).
            **kwargs: Additional keyword arguments for configuration (e.g., budget tokens, device).

        Example:
            ```python
            model = EchoRankReranker(method='echorank', model_name='flan-t5-large')
            ```
        """
        self.method = method
        self.model_name = model_name or kwargs.get("model_name", "google/flan-t5-large")
        self.type = kwargs.get("type", "cheap")
        self.budget_tokens = kwargs.get("budget_tokens", 4000)
        self.budget_split_x = kwargs.get("budget_split_x", 0.5)
        self.budget_split_y = kwargs.get("budget_split_y", 0.5)
        self.total_passages = kwargs.get("total_passages", 50)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model using Hugging Face pipeline
        self.model = pipeline("text2text-generation", model=self.model_name, device=self.device)

    def _get_binary_response(self, passage: str, query: str) -> str:
        """
        Determines binary relevance ("yes" or "no") of a passage relative to the query.

        Args:
            passage (str): The passage text.
            query (str): The query text.

        Returns:
            str: "yes" or "no" indicating relevance.

        Example:
            ```python
            response = model._get_binary_response(passage, query)
            ```
        """
        prompt = f"Is the following passage related to the query?\npassage: {passage}\nquery: {query}\nAnswer in yes or no."
        return self.model(prompt)[0]["generated_text"].strip().lower()

    def _get_pairwise_response(self, query: str, passage_a: str, passage_b: str) -> str:
        """
        Compares two passages and selects the one more relevant to the query.

        Args:
            query (str): The query text.
            passage_a (str): First passage.
            passage_b (str): Second passage.

        Returns:
            str: "passage a" or "passage b" indicating which passage is more relevant.

        Example:
            ```python
            better_passage = model._get_pairwise_response(query, passage_a, passage_b)
            ```
        """
        prompt = f"""Given a query "{query}", which of the following two passages is more relevant to the query?
Passage A: {passage_a}
Passage B: {passage_b}
Output Passage A or Passage B."""
        return self.model(prompt)[0]["generated_text"].strip().lower()

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks contexts within each document using the EchoRank two-stage process.

        Args:
            documents (List[Document]): Documents containing queries and associated contexts.

        Returns:
            List[Document]: Documents updated with reordered contexts after EchoRank reranking.

        Raises:
            ValueError: If any document lacks contexts for reranking.

        Example:
            ```python
            reranked_docs = model.rank(documents)
            ```
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            contexts = document.contexts

            # Stage 1: Binary Classification
            binary_token_limit = int(self.budget_split_x * self.budget_tokens)
            binary_running_token = 0
            yes_contexts, no_contexts = [], []

            for context in contexts[:self.total_passages]:
                text = context.text
                token_length = len(text.split())
                if binary_running_token + token_length < binary_token_limit:
                    response = self._get_binary_response(text, query)
                    if "yes" in response:
                        yes_contexts.append(context)
                    else:
                        no_contexts.append(context)
                    binary_running_token += token_length

            # Stage 2: Pairwise Ranking
            pairwise_contexts = yes_contexts[:int(self.budget_split_y * self.budget_tokens)]
            for i in range(len(pairwise_contexts) - 1):
                for j in range(i + 1, len(pairwise_contexts)):
                    response = self._get_pairwise_response(query, pairwise_contexts[i].text, pairwise_contexts[j].text)
                    if response == "passage b":
                        pairwise_contexts[i], pairwise_contexts[j] = pairwise_contexts[j], pairwise_contexts[i]

            # Assign scores and sort
            all_contexts = copy.deepcopy(pairwise_contexts + no_contexts )
            for context in all_contexts:
                context.score = 1.0 if context in pairwise_contexts else 0.0  # Assigning scores based on pairwise ranking
            
            ranked_contexts = sorted(all_contexts, key=lambda ctx: ctx.score, reverse=True)

            # Update Document with reranked contexts
            document.reorder_contexts = ranked_contexts

        return documents

