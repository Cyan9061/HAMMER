
import hashlib
import typing as T
from abc import ABC, abstractmethod

import datasets
import pandas as pd
from datasets import Features, Sequence, Value
from llama_index.core import Document
from overrides import overrides
from pydantic import BaseModel

from hammer.configuration import cfg
from hammer.utils.locks import distributed_lock
from hammer.logger import logger

class QAPair(BaseModel):
    question: str
    answer: str
    id: str
    context: T.Union[T.Dict[str, str], T.List[T.Dict]]
    supporting_facts: T.List[T.Any]
    difficulty: str
    qtype: str
    
    # 🔥 新增字段支持统一数据集格式
    text_ground_truth: T.List[str] = []  # 支撑文档列表
    dataset_name: str = ""  # 数据集名称标识
    metadata: T.Dict[str, T.Any] = {}  # 额外元数据

class PartitionMap(BaseModel):
    sample: str = "sample"
    train: str = "train"
    test: str = "test"
    holdout: str = "holdout"

class HammerQADataset(BaseModel, ABC):
    """Container and utilities for dataset with remote storage.

    Instances of this class do _not_ store the dataset; only pointers to it.
    Therefore, this class can be safely passed to Ray tune.
    """

    # Canonical name for the dataset
    # Set this to a unique string Literal on your subclass
    # Must be present when loading a StudyConfig from a yaml
    xname: T.Literal["HammerQADataset"] = "HammerQADataset"

    # Partition names for this dataset as it is stored on disk
    storage_partitions: T.List[str] = ["sample", "train", "test", "holdout"]
    # How to map requested partition to the storage partitions
    # eg. MyDataset(partition_map={'test': 'sample'}) to run on the sample partition
    partition_map: PartitionMap = PartitionMap()

    # timeouts
    load_examples_timeout_s: int = 3600
    load_grounding_data_timeout_s: int = 3600

    @property
    def name(self) -> str:
        """Subclasses may dynamically construct name."""
        return self.xname

    def _get_storage_partition(self, canonical_partition: str) -> str:
        return getattr(self.partition_map, canonical_partition)

    @abstractmethod
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        pass

    @abstractmethod
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        pass

class InfiniteBenchHF(HammerQADataset):
    xname: T.Literal["infinitebench_hf"] = "infinitebench_hf"  # type: ignore
    subset: T.Literal["longbook_qa_eng"] = "longbook_qa_eng"
    description: str = "The dataset contains a large number of books."

    def _load_raw_dataset(self) -> pd.DataFrame:
        ft = Features(
            {
                "id": Value("int64"),
                "context": Value("string"),
                "input": Value("string"),
                "answer": Sequence(Value("string")),
                "options": Sequence(Value("string")),
            }
        )

        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "xinrongzhang2022/InfiniteBench",
                features=ft,
                split=self.subset,
                cache_dir=cfg.paths.huggingface_cache,
            )
        df = dataset.to_pandas()
        return df

    def _add_partitions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["book_id"] = df.context.factorize()[0]
        books = df.book_id.unique()
        book_partitions = {
            "sample": books[:1],
            "train": books[1:23],
            "test": books[23:46],
            "holdout": books[46:69],
        }

        def label_partition(book_id: int) -> str:
            for partition, book_range in book_partitions.items():
                if book_id in book_range:
                    return partition
            raise IndexError(f"Book id {book_id} is out of range of {book_partitions=}")

        df["partition"] = df.book_id.apply(label_partition)
        return df

    @property
    def _dataset(self) -> pd.DataFrame:
        df = self._load_raw_dataset()
        df = self._add_partitions(df)
        return df

    def _row_to_qapair(self, row: T.Dict[str, T.Any]) -> QAPair:
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["input"],
            answer=str(row["answer"]),
            id=str(row["id"]),
            context={"book_start": row["context"][:100]},
            supporting_facts=[],
            difficulty="",
            qtype="",
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        df = self._dataset
        partition = self._get_storage_partition(partition)
        df = df[df.partition == partition]
        for _, row in df.iterrows():
            yield self._row_to_qapair(row)

    @overrides
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        df = self._dataset
        partition = self._get_storage_partition(partition)
        df = df[df.partition == partition]
        for book in df.context.unique():
            yield Document(
                text=book,
            )

class FinanceBenchHF(HammerQADataset):
    xname: T.Literal["financebench_hf"] = "financebench_hf"  # type: ignore
    description: str = (
        "Financial dataset that contains everything about finance, including real-world financial documents, "
        "SEC filings, earning reports, call transcripts, and much more. "
        "It has all the financial live data, historical data, just about everything about finance, for instance, "
        "definitions and explanations of financial term, "
        "insights on company revenues, mergers, founders, or stock performance, "
        "details on financial laws, compliance, or government policies, "
        "information required to evaluated finance risk, and "
        "information about banking operations, credit systems, or loan structures."
    )

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/financebench",
                "groundingdata",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/financebench",
                "qapairs",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    @overrides
    def iter_grounding_data(
        self, partition="test", **load_kwargs
    ) -> T.Iterator[Document]:
        assert partition in self.storage_partitions
        grounding_dataset = self._load_grounding_dataset()
        partition = self._get_storage_partition(partition)
        for row in grounding_dataset[partition]:
            yield Document(
                text=row["html"],
                metadata={"file_name": row["filename"]},
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["question"],
            answer=row["answer"],
            id=str(row["financebench_id"]),
            context=row["evidence"],
            supporting_facts=[row["justification"]],
            difficulty="",
            qtype=row["question_type"],
            gold_evidence=row["html_evidence"],
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        for row in qa_examples[partition]:
            yield self._row_to_qapair(row)

class SyntheticFinanceBenchHF(FinanceBenchHF):
    xname: T.Literal["synthetic_financebench_hf"] = "synthetic_financebench_hf"  # type: ignore

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/financebench",
                "qapairs_synthetic",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["query"],
            answer=row["reference_answer"],
            id=str(row["id"]),
            context={},
            supporting_facts=[],
            difficulty="default",
            qtype="default",
        )

class HotPotQAHF(HammerQADataset):
    xname: T.Literal["hotpotqa_hf"] = "hotpotqa_hf"  # type: ignore
    subset: str = "dev"  # train_hard, dev
    description: str = (
        "This dataset is a vast collection of all kind of information that you can find on Wikipedia. "
        "It can be used, for instance, to retrieve straightforward facts from one or more documents, "
        "compare two entities based on shared attributes, "
        "identify relationships, roles, or attributes of entities, "
        "reason about dates, timelines, or chronological order, "
        "determine geographical relationships or locations, "
        "explain causes or sequences of events or processes, "
        "synthesize facts from multiple documents to infer answers, and "
        "validate or refute premises in the context of the question."
    )

    @property
    def name(self) -> str:
        return f"{self.xname}/{self.subset}"

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/hotpotqa",
                f"groundingdata_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/hotpotqa",
                f"qapairs_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    @overrides
    def iter_grounding_data(
        self,
        partition="test",
    ) -> T.Iterator[Document]:
        assert partition in self.storage_partitions
        grounding_dataset = self._load_grounding_dataset()
        partition = self._get_storage_partition(partition)
        for row in grounding_dataset[partition]:
            yield Document(
                text=row["text"],
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["question"],
            answer=row["answer"],
            id=str(row["id"]),
            context=[{title: sentence} for title, sentence in row["context"]],
            supporting_facts=row["supporting_facts"],
            difficulty=row["level"],
            qtype=row["type"],
            gold_evidence=[sentence for _, sentence in row["context"]],
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        for row in qa_examples[partition]:
            yield self._row_to_qapair(row)

class SyntheticHotPotQAHF(HotPotQAHF):
    xname: T.Literal["synthetic_hotpotqa_hf"] = "synthetic_hotpotqa_hf"  # type: ignore

    @property
    def name(self) -> str:
        return f"{self.xname}/{self.subset}"

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/hotpotqa",
                f"qapairs_synthetic_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["query"],
            answer=row["reference_answer"],
            id=str(row["id"]),
            context={},
            supporting_facts=[],
            difficulty="default",
            qtype="default",
        )

class CragTask3HF(HammerQADataset):
    xname: T.Literal["crag_hf"] = "crag_hf"  # type: ignore
    subset: str = "sports"  # finance, movie, music, sports, open

    _descriptions = {
        "sports": (
            "This resource contains everything about sports, including sports news, sports events, "
            "sports statistics, sports schedules, and more. "
            "It contains historical data, live data, just about everything about sports, for instance, "
            "information about athlete achievements, teams, or career stats, "
            "information on match dates, scores, or tournaments, "
            "team performances or player statistics, "
            "key events or records in sports history, and "
            "information about rules or formats in specific sports."
        ),
        "finance": (
            "This resource contains everything about finance, including financial news, "
            "market data, financial reports, and more. "
            "It contains historical data, live data, just about everything about finance, for instance, "
            "stock prices, trends, or market indices, "
            "revenue, founders, or headquarters of companies, "
            "major financial events or policy changes, "
            "comparing returns or risks of different investments, and "
            "financial terms or concepts explained."
        ),
        "movie": (
            "This resource contains everything about movies, including movie news, reviews, "
            "box office data, and more. "
            "It contains historical data, live data, just about everything about movies, for instance, "
            "information about actors, directors, or crew roles, "
            "movies by a specific actor or director, "
            "key events or summaries from movie plots, "
            "recognition received by films or individuals, and "
            "dates or production details for movies."
        ),
        "music": (
            "This resource contains everything about music, including music news, charts, "
            "artist information, and more. "
            "It contains historical data, live data, just about everything about music, for instance, "
            "information on musicians, albums, or band members, "
            "albums or songs released by specific artists, "
            "information about song lyrics or their significance, "
            "Grammys or other recognitions for artists or albums, and "
            "details about music styles or trends over time."
        ),
        "open": (
            "This resource contains everything about open domain, including general knowledge, "
            "trivia, fun facts, and more. "
            "It contains historical data, live data, just about everything about open domain, for instance, "
            "general knowledge about diverse topics, "
            "facts that combine multiple subject areas, "
            "unique or interesting facts across domains (trivia), and "
            "further material useful for complex, unconstrained reasoning without a specific domain."
        ),
    }

    @property
    def name(self) -> str:
        return f"{self.xname}/{self.subset}"

    @property
    def description(self) -> str:
        return self._descriptions[self.subset]

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/crag",
                f"groundingdata_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/crag",
                f"qapairs_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    @overrides
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        assert partition in self.storage_partitions
        grounding_dataset = self._load_grounding_dataset()
        partition = self._get_storage_partition(partition)
        for row in grounding_dataset[partition]:
            yield Document(
                text=row["markdown"],
                metadata={"file_name": row["filename"]},
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        context = {result["_hash"]: str(result) for result in row["search_results"]}
        return QAPair(
            question=row["query"],
            answer=row["answer"],
            id=str(row["interaction_id"]),
            context=context,
            supporting_facts=[],
            difficulty="default",
            qtype=row["question_type"],
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        for row in qa_examples[partition]:
            yield self._row_to_qapair(row)

class SyntheticCragTask3HF(CragTask3HF):
    xname: T.Literal["synthetic_crag_hf"] = "synthetic_crag_hf"  # type: ignore

    @property
    def name(self) -> str:
        return f"{self.xname}/{self.subset}"

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/crag",
                f"qapairs_synthetic_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["query"],
            answer=row["reference_answer"],
            id=str(row["id"]),
            context={},
            supporting_facts=[],
            difficulty="default",
            qtype="default",
        )

class DRDocsHF(HammerQADataset):
    xname: T.Literal["drdocs_hf"] = "drdocs_hf"  # type: ignore
    description: str = (
        "The dataset contains comprehensive information about DataRobot, including its API, documentation, examples, "
        "key features, platform architecture, integrations, setup guides, data handling, feature engineering, EDA tools, "
        "automated machine learning, model management, deployment options, monitoring, REST API, batch predictions, "
        "real-time scoring, custom recipes, retraining, lifecycle management, bias detection, explainability, diagnostics, "
        "cross-validation, leaderboard insights, time series modeling, data governance, security, user roles, Python/R usage, "
        "custom blueprints, external model integration, Docker deployments, API reference, BI tool integration, workflow automation, "
        "multimodal modeling, NLP, image recognition, hyperparameter tuning, performance optimization, resource management, "
        "parallel processing, drift detection, retraining triggers, industry use cases, tutorials, case studies, common issues, "
        "debugging tips, FAQs, support access, community resources, and release notes."
    )

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/drdocs",
                "groundingdata",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/drdocs",
                "qapairs",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        # There is no partition. The grounding dataset is the same
        # across all partitions of the qa pairs.
        grounding_dataset = self._load_grounding_dataset()
        for row in grounding_dataset["train"]:
            yield Document(
                text=row["markdown"],
                metadata={"file_name": row["filename"]},
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["question"],
            answer=row["answer"],
            id=str(row["id"]),
            context={},
            supporting_facts=[],
            difficulty="default",
            qtype="default",
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        assert partition in self.storage_partitions
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        for row in qa_examples[partition]:
            yield self._row_to_qapair(row)

class PhantomWikiv050(HammerQADataset):
    xname: T.Literal["phantomwikiv050_hf"] = "phantomwikiv050_hf"  # type: ignore

    # subset choices are:
    # 'depth_20_size_25_seed_1',
    # 'depth_20_size_25_seed_2',
    # 'depth_20_size_25_seed_3',
    # 'depth_20_size_50_seed_1',
    # 'depth_20_size_50_seed_2',
    # 'depth_20_size_50_seed_3',
    # 'depth_20_size_100_seed_1',
    # 'depth_20_size_100_seed_2',
    # 'depth_20_size_100_seed_3',
    # 'depth_20_size_200_seed_1',
    # 'depth_20_size_200_seed_2',
    # 'depth_20_size_200_seed_3',
    # 'depth_20_size_300_seed_1',
    # 'depth_20_size_300_seed_2',
    # 'depth_20_size_300_seed_3',
    # 'depth_20_size_400_seed_1',
    # 'depth_20_size_400_seed_2',
    # 'depth_20_size_400_seed_3',
    # 'depth_20_size_500_seed_1',
    # 'depth_20_size_500_seed_2',
    # 'depth_20_size_500_seed_3',
    # 'depth_20_size_1000_seed_1',
    # 'depth_20_size_1000_seed_2',
    # 'depth_20_size_1000_seed_3',
    # 'depth_20_size_2500_seed_1',
    # 'depth_20_size_2500_seed_2',
    # 'depth_20_size_2500_seed_3',
    # 'depth_20_size_5000_seed_1',
    # 'depth_20_size_5000_seed_2',
    # 'depth_20_size_5000_seed_3',
    # 'depth_20_size_10000_seed_1',
    # 'depth_20_size_10000_seed_2',
    # 'depth_20_size_10000_seed_3'

    subset: str = "depth_20_size_10000_seed_3"
    description: str = (
        "This dataset contains data from PhantomWiki, which "
        "is a framework for generating unique, factually"
        "and consistent document corpora with diverse question-answer pairs."
        "Unlike prior work, PhantomWiki is neither a fixed dataset, nor is it"
        "based on any existing data. Instead, a new PhantomWiki instance is "
        "generated on demand for each evaluation. PhantomWiki generates a "
        "fictional universe of characters along with a set of facts. "
        "We reflect these facts in a large-scale corpus, mimicking the "
        "style of fan-wiki websites. Then we generate question-answer pairs "
        "with tunable difficulties, encapsulating the types of multi-hop "
        "questions commonly considered in the question-answering (QA) literature."
    )

    def _get_partition_range(self, partition: str):
        partition_ranges = {
            "sample": range(0, 10),
            "train": range(0, 100),
            "test": range(100, 400),
            "holdout": range(400, 500),
        }
        if partition in partition_ranges:
            return partition_ranges[partition]

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "kilian-group/phantom-wiki-v050",
                "text-corpus",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset[self.subset]

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "kilian-group/phantom-wiki-v050",
                "question-answer",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset[self.subset]

    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        # There is no partition. The grounding dataset is the same
        # across all partitions of the qa pairs.
        grounding_dataset = self._load_grounding_dataset()
        for row in grounding_dataset:
            yield Document(
                text=row["article"],
                metadata={"title": row["title"]},
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["question"],
            answer=" ".join(row["answer"]),
            id=str(row["id"]),
            context={},
            supporting_facts=[],
            difficulty=str(row["difficulty"]),
            qtype=str(row["type"]),
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        assert partition in self.storage_partitions
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        partition_range = self._get_partition_range(partition)

        for i in partition_range:
            row = qa_examples[i]
            yield self._row_to_qapair(row)

class MultiHopRAGHF(HammerQADataset):
    xname: T.Literal["multihoprag_hf"] = "multihoprag_hf"  # type: ignore

    description: str = (
        "This resource contains a corpus of news dataset. "
        "The news data source comprises various English-language "
        "websites covering a range of news categories: "
        "entertainment, business, sports, technology, "
        "health and science. It contains news articles "
        "on these topics published between September to December 2023. "
        "Every news article is paired with metadata, "
        "including title, publish date, author, category, URL and news source."
    )

    def _get_partition_range(self, partition: str):
        partition_ranges = {
            "sample": range(0, 20),
            "train": range(0, 1000),
            "test": range(1000, 2000),
            "holdout": range(2000, 2556),
        }
        if partition in partition_ranges:
            return partition_ranges[partition]

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "yixuantt/MultiHopRAG",
                "corpus",
                cache_dir=cfg.paths.huggingface_cache,
            )
        # all of the corpus is in a split named "train"
        return dataset["train"]

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "yixuantt/MultiHopRAG",
                "MultiHopRAG",
                cache_dir=cfg.paths.huggingface_cache,
            )

        # all qapairs are in a split named "train"
        return dataset["train"]

    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        # There is no partition. The grounding dataset is the same
        # across all partitions of the qa pairs.
        grounding_dataset = self._load_grounding_dataset()
        for row in grounding_dataset:
            yield Document(
                text=row["body"],
                metadata={
                    "title": row["title"],
                    "author": row["author"],
                    "category": row["category"],
                    "published_at": row["published_at"],
                    "url": row["url"],
                    "source": row["source"],
                },
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["query"],
            answer=row["answer"],
            id=hashlib.md5(row["query"].encode()).hexdigest(),
            context={},
            supporting_facts=row["evidence_list"],
            difficulty="",
            qtype=str(row["question_type"]),
            gold_evidence=[item["fact"] for item in row["evidence_list"]],
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        assert partition in self.storage_partitions
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        partition_range = self._get_partition_range(partition)

        for i in partition_range:
            row = qa_examples[i]
            yield self._row_to_qapair(row)

class BrightHF(HammerQADataset):
    xname: T.Literal["bright_hf"] = "bright_hf"  # type: ignore

    # NOTE: pony, leetcode, aops, theoremqa_theorems, theoremqa_questions
    # subsets do not have ground truth answers
    subset: T.Literal[
        "earth_science",
        "biology",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "pony",
        "leetcode",
        "aops",
        "theoremqa_theorems",
        "theoremqa_questions",
    ] = "biology"

    _descriptions = {
        "earth_science": (
            "This resource contains everything about earth science, including geology, "
            "oceanography, meteorology, and more. "
            "It contains historical data, live data, just about everything about earth science, for instance, "
            "information about geological formations, ocean currents, or weather patterns, "
            "key events or phenomena in earth science history, "
            "information about climate change or environmental issues, and "
            "information about earth science research methods or techniques."
        ),
        "biology": (
            "This resource contains everything about biology, including genetics, "
            "ecology, evolution, and more. "
            "It contains historical data, live data, just about everything about biology, for instance, "
            "information about cellular structures, ecosystems, or evolutionary processes, "
            "key events or discoveries in biology history, "
            "information about genetic engineering or biotechnology, and "
            "information about biological research methods or techniques."
        ),
        "economics": (
            "This resource contains everything about economics, including microeconomics, "
            "macroeconomics, international trade, and more. "
            "It contains historical data, live data, just about everything about economics, for instance, "
            "information about supply and demand, market structures, or economic indicators, "
            "key events or theories in economics history, "
            "information about economic policies or regulations, and "
            "information about economic research methods or techniques."
        ),
        "psychology": (
            "This resource contains everything about psychology, including cognitive, "
            "developmental, social, and clinical psychology. "
            "It contains historical data, live data, just about everything about psychology, for instance, "
            "information about brain structures, psychological disorders, or therapeutic techniques, "
            "key events or theories in psychology history, "
            "information about psychological research methods or techniques, and "
            "information about psychological assessments or tests."
        ),
        "robotics": (
            "This resource contains everything about robotics, including robot design, "
            "control systems, and applications. "
            "It contains historical data, live data, just about everything about robotics, for instance, "
            "information about robot components, sensors, or actuators, "
            "key events or advancements in robotics history, "
            "information about robotic applications in various fields, and "
            "information about robotics research methods or techniques."
        ),
        "stackoverflow": (
            "This resource contains everything about StackOverflow, including programming, "
            "software development, and more. "
            "It contains historical data, live data, just about everything about StackOverflow, for instance, "
            "information about programming languages, software frameworks, or development methodologies, "
            "key events or trends in software development history, "
            "information about software development best practices or tools, and "
            "information about software development research methods or techniques."
        ),
        "sustainable_living": (
            "This resource contains everything about sustainable living, including "
            "environmental conservation, renewable energy, and more. "
            "It contains historical data, live data, just about everything about sustainable living, for instance, "
            "information about sustainable practices, technologies, or policies, "
            "key events or movements in sustainability history, "
            "information about environmental issues or challenges, and "
            "information about sustainability research methods or techniques."
        ),
        "pony": (
            "This resource contains everything about the Pony programming language, including its syntax, "
            "capabilities, compiler architecture, and more. "
            "It contains documentation, examples, just about everything about Pony, for instance, "
            "information about Pony's actor-based concurrency model, capabilities-secure type system, "
            "key concepts like reference capabilities and behavior guarantees, "
            "information about pattern matching, generics, and other language features, and "
            "information about Pony's memory management, garbage collection, and runtime performance."
        ),
        "leetcode": (
            "This resource contains everything about LeetCode, including its problems, "
            "solutions, and more. "
            "It contains documentation, examples, just about everything about LeetCode, for instance, "
            "information about data structures, algorithms, or coding challenges, "
            "key concepts like time complexity and space complexity, "
            "information about problem-solving techniques and strategies, and "
            "information about LeetCode's online judge system and coding environment."
        ),
        "aops": (
            "This resource contains everything about Art of Problem Solving (AoPS), including its curriculum, "
            "problems, and more. "
            "It contains documentation, examples, just about everything about AoPS, for instance, "
            "information about problem-solving techniques, mathematical concepts, or competition preparation, "
            "key events or competitions in mathematics history, "
            "information about AoPS's online community and resources, and "
            "information about AoPS's books and courses."
        ),
        "theoremqa_theorems": (
            "This resource contains everything about TheoremQA, including its theorems, "
            "proofs, and more. "
            "It contains documentation, examples, just about everything about TheoremQA, for instance, "
            "information about mathematical theorems, proofs, or concepts, "
            "key events or advancements in theorem proving history, "
            "information about theorem proving techniques and strategies, and "
            "information about TheoremQA's online community and resources."
        ),
        "theoremqa_questions": (
            "This resource contains everything about TheoremQA, including its questions, "
            "answers, and more. "
            "It contains documentation, examples, just about everything about TheoremQA, for instance, "
            "information about mathematical questions, answers, or concepts, "
            "key events or advancements in theorem proving history, "
            "information about theorem proving techniques and strategies, and "
            "information about TheoremQA's online community and resources."
        ),
    }

    @property
    def description(self) -> str:
        return self._descriptions[self.subset]

    def _get_partition_range(self, partition: str):
        subset_partition_ranges = {
            "earth_science": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 116),
            },
            "biology": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 103),
            },
            "economics": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 103),
            },
            "psychology": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 101),
            },
            "robotics": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 101),
            },
            "stackoverflow": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 117),
            },
            "sustainable_living": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 108),
            },
            "pony": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 112),
            },
            "leetcode": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 120),
                "holdout": range(120, 142),
            },
            "aops": {
                "sample": range(0, 10),
                "train": range(0, 30),
                "test": range(30, 90),
                "holdout": range(90, 111),
            },
            "theoremqa_theorems": {
                "sample": range(0, 10),
                "train": range(0, 20),
                "test": range(20, 50),
                "holdout": range(50, 76),
            },
            "theoremqa_questions": {
                "sample": range(0, 10),
                "train": range(0, 40),
                "test": range(40, 160),
                "holdout": range(160, 194),
            },
        }

        partition_ranges = subset_partition_ranges[self.subset]
        if partition in partition_ranges:
            return partition_ranges[partition]

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "xlangai/BRIGHT",
                "documents",
                cache_dir=cfg.paths.huggingface_cache,
            )
        # all of the corpus is in a split named "train"
        return dataset[self.subset]

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "xlangai/BRIGHT",
                "examples",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset[self.subset]

    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        # There is no partition. The grounding dataset is the same
        # across all partitions of the qa pairs.
        grounding_dataset = self._load_grounding_dataset()
        for row in grounding_dataset:
            yield Document(
                text=row["content"],
                metadata={
                    "id": row["id"],
                },
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["query"],
            answer=row["gold_answer"],
            id=row["id"],
            context={},
            supporting_facts=[],
            difficulty="",
            qtype="",
        )

    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        assert partition in self.storage_partitions
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        partition_range = self._get_partition_range(partition)

        for i in partition_range:
            row = qa_examples[i]
            yield self._row_to_qapair(row)

class TwoWikiMultiHopQA(HammerQADataset):
    """2WikiMultiHopQA dataset for multi-hop reasoning evaluation."""
    xname: T.Literal["2wikimultihopqa"] = "2wikimultihopqa"  # type: ignore
    
    description: str = (
        "2WikiMultiHopQA is a multi-hop QA dataset that uses structured and unstructured data. "
        "It aims to test reasoning and inference skills by requiring a model to read multiple "
        "paragraphs to answer a given question. The dataset includes evidence information "
        "containing a reasoning path for multi-hop questions."
    )
    
    # Declare fields for Pydantic model
    corpus_file: str = "docs/2wikimultihopqa_corpus.json"
    qa_file: str = "docs/2wikimultihopqa.json"
        
    def _load_corpus_data(self) -> T.List[T.Dict[str, str]]:
        """Load the corpus documents from local JSON file."""
        import json
        from pathlib import Path
        
        corpus_path = Path(self.corpus_file)
        if not corpus_path.is_absolute():
            # Look for file in project root
            corpus_path = Path(__file__).parent.parent / self.corpus_file
            
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"Corpus file not found: {corpus_path}")
            return []
    
    def _load_qa_data(self) -> T.List[T.Dict[str, T.Any]]:
        """Load the QA pairs from local JSON file."""
        import json
        from pathlib import Path
        
        qa_path = Path(self.qa_file)
        if not qa_path.is_absolute():
            # Look for file in project root
            qa_path = Path(__file__).parent.parent / self.qa_file
            
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"QA file not found: {qa_path}")
            return []
    
    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        """Iterate over corpus documents for grounding."""
        corpus_data = self._load_corpus_data()
        for doc in corpus_data:
            yield Document(
                text=doc["text"],
                metadata={"title": doc["title"]},
            )
    
    def _row_to_qapair(self, row: T.Dict[str, T.Any]) -> QAPair:
        """Convert 2WikiMultiHopQA row to QAPair."""
        # Extract context as flattened dictionary for compatibility
        context = {}
        if "context" in row:
            for title, sentences in row["context"]:
                context[title] = " ".join(sentences) if isinstance(sentences, list) else sentences
        
        # Format supporting facts 
        supporting_facts = row.get("supporting_facts", [])
        
        # Extract evidences for multi-hop reasoning
        evidences = row.get("evidences", [])
        
        # Determine hop type from question type
        hop_type = row.get("type", "single")
        if hop_type in ["comparison", "inference", "compositional", "bridge-comparison"]:
            hop_type = hop_type
        else:
            hop_type = "single"
        
        return QAPair(
            question=row["question"],
            answer=row["answer"],
            id=str(row["_id"]),
            context=context,
            supporting_facts=supporting_facts,
            difficulty=row.get("level", "medium"),
            qtype=row.get("type", "multi-hop"),
            evidences=evidences,
            hop_type=hop_type
        )
    
    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        """Iterate over QA examples."""
        qa_data = self._load_qa_data()
        
        # For simplicity, use all data for any partition
        # In a real implementation, you would split by partition
        for row in qa_data:
            yield self._row_to_qapair(row)

class UnifiedJSONDataset(HammerQADataset):
    """🔥 统一JSON格式数据集加载器 - 支持6个标准化数据集"""
    
    xname: T.Literal["unified_json"] = "unified_json"  
    dataset_name: str = "2wikimultihopqa"  # 数据集名称
    base_path: str = "docs/dataset/unified"
    
    # 🔑 修复：使用ClassVar避免Pydantic字段错误
    SUPPORTED_DATASETS: T.ClassVar[T.List[str]] = [
        "2wikimultihopqa", "hotpotqa", "musique",  # 多跳推理数据集
        "FinQA", "MedQA", "bioasq"  # 领域专业数据集  
    ]
    
    @property
    def name(self) -> str:
        return f"{self.xname}_{self.dataset_name}"
    
    def _load_corpus_data(self) -> T.Dict[str, str]:
        """加载语料库数据"""
        import json
        corpus_file = f"{self.base_path}/{self.dataset_name}_corpus_unified.json"
        corpus_dict = {}
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        corpus_dict[data["id"]] = data["text"]
            logger.info(f"✅ 加载语料库: {len(corpus_dict)} 条文档 from {corpus_file}")
        except Exception as e:
            logger.error(f"❌ 加载语料库失败: {corpus_file}, 错误: {e}")
        return corpus_dict
    
    def _load_qa_data(self) -> T.List[T.Dict]:
        """加载QA数据"""
        import json
        qa_file = f"{self.base_path}/{self.dataset_name}_qa_unified.json"
        qa_data = []
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        qa_data.append(json.loads(line.strip()))
            logger.info(f"✅ 加载QA数据: {len(qa_data)} 条问题 from {qa_file}")
        except Exception as e:
            logger.error(f"❌ 加载QA数据失败: {qa_file}, 错误: {e}")
        return qa_data
    
    @overrides
    def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
        """迭代语料库文档"""
        corpus_data = self._load_corpus_data()
        for doc_id, text in corpus_data.items():
            yield Document(
                text=text,
                metadata={"id": doc_id, "dataset": self.dataset_name}
            )
    
    def _row_to_qapair(self, row: T.Dict) -> QAPair:
        """转换为标准QAPair格式"""
        # 🔧 修复：将text_ground_truth字符串列表转换为QAPair期望的字典列表格式
        text_ground_truth = row.get("text_ground_truth", [])
        if isinstance(text_ground_truth, list) and text_ground_truth:
            # 将字符串列表转换为字典列表格式，符合QAPair.context的类型要求
            if isinstance(text_ground_truth[0], str):
                context_formatted = [{"entity": entity} for entity in text_ground_truth]
            else:
                context_formatted = text_ground_truth  # 已经是字典列表格式
        else:
            context_formatted = []
        
        return QAPair(
            question=row["query"],
            answer=row["answer_ground_truth"],
            id=row["id"],
            context=context_formatted,  # 🔧 使用格式化后的context
            supporting_facts=row.get("metadata", {}).get("supporting_facts", []),
            difficulty=row.get("metadata", {}).get("difficulty", "medium"),
            qtype=row.get("metadata", {}).get("type", "unknown"),
            text_ground_truth=row.get("text_ground_truth", []),  # 🔥 新字段
            dataset_name=self.dataset_name,  # 🔥 新字段
            metadata=row.get("metadata", {})  # 🔥 新字段
        )
    
    @overrides
    def iter_examples(self, partition="test") -> T.Iterator[QAPair]:
        """迭代QA样例"""
        qa_data = self._load_qa_data()
        
        # 简化分区逻辑：对于统一数据集，使用全部数据
        # 在实际应用中，可以根据需求进行数据划分
        for row in qa_data:
            yield self._row_to_qapair(row)

