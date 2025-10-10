import numpy as np
import faiss
import random
import networkx as nx
from tqdm import tqdm
from typing import List, Any
from types import SimpleNamespace
import json
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
# Import QueryBundle for the new retrieve API
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class L2FaissSelector:
    """
    Uses LlamaIndex and Faiss (HNSW, L2 distance) to build an optimized "dissimilarity" graph,
    and applies graph algorithms to select a diverse subset of queries.
    """
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        print(f"Loading embedding model: {model_name}...")
        self.model = HuggingFaceEmbedding(model_name=model_name)
        Settings.embed_model = self.model
        self.embeddings = None
        self.index = None

    def _build_index(self, dataset: List[SimpleNamespace]):
        """
        Builds the index using a robust, explicit embedding workflow that aligns with best practices.
        """
        if not dataset:
            print("Error: Dataset is empty, cannot build index.")
            return

        print("Step 1: Explicitly generating embeddings for all texts...")
        texts = [data.question for data in dataset]
        # 1. 显式地、一次性地为所有文本生成嵌入向量
        embeddings_list = self.model.get_text_embedding_batch(texts, show_progress=True)
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        
        d = self.embeddings.shape[1]
        print(f"Embedding generation complete. Dimension: {d}")

        print("Step 2: Building Faiss index and LlamaIndex components...")
        # 2. 像之前一样准备 Faiss 索引和 LlamaIndex 的存储组件
        faiss_index = faiss.IndexHNSWFlat(d, 32)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 3. 创建一个空的 VectorStoreIndex 作为容器
        # 注意：这里不再传入 nodes，因为我们将手动插入它们
        self.index = VectorStoreIndex(
            [],
            storage_context=storage_context,
        )

        print("Step 3: Creating and inserting nodes with pre-computed embeddings...")
        # 4. 创建带有预计算嵌入的 TextNode
        nodes = []
        for i, data in enumerate(dataset):
            node = TextNode(
                id_=str(i),
                text=data.question,
                embedding=self.embeddings[i].tolist()  # 在创建节点时，使用 list 更安全
            )
            nodes.append(node)
        
        # 5. 将包含嵌入的完整节点批量插入索引
        self.index.insert_nodes(nodes, show_progress=True)

        print("L2 distance index built successfully using a robust workflow.")

    def _build_dissimilarity_graph(self, dataset: List[SimpleNamespace], similarity_threshold: float, num_samples_per_node: int, exclude_top_k: int):
        """
        Builds the dissimilarity graph using cosine similarity.
        Nodes are connected if their cosine similarity is **less than similarity_threshold**.
        (That is, they're dissimilar enough.)
        """
        if self.index is None or self.embeddings is None:
            raise ValueError("Index and embeddings must be built first.")

        num_nodes = len(self.embeddings)
        print(f"Building dissimilarity graph (cosine_sim < {similarity_threshold}, samples_per_node={num_samples_per_node})...")

        graph = {i: {} for i in range(num_nodes)}

        retriever = self.index.as_retriever(
            similarity_top_k=exclude_top_k,
            embed_model=self.model
        )

        # Step 1: Normalize all embeddings to unit vectors for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normed_embeddings = self.embeddings / (norms + 1e-10)  # avoid division by zero

        for i in tqdm(range(num_nodes), desc="Building dissimilarity graph (Cosine Sim)"):
            query_embedding = normed_embeddings[i]
            query_text = dataset[i].question

            query_bundle = QueryBundle(query_str=query_text, embedding=query_embedding)
            similar_results = retriever.retrieve(query_bundle)
            similar_indices = {int(res.node.node_id) for res in similar_results}
            similar_indices.add(i)

            all_indices = set(range(num_nodes))
            pool_indices = list(all_indices - similar_indices)
            if not pool_indices:
                continue

            sampled_indices = random.sample(pool_indices, min(num_samples_per_node, len(pool_indices)))
            if not sampled_indices:
                continue

            sampled_embeddings = normed_embeddings[sampled_indices]  # shape: (sample_count, d)

            # Cosine similarity as dot product (since normalized)
            cosine_similarities = query_embedding @ sampled_embeddings.T  # shape: (sample_count,)

            for j_local, j_global in enumerate(sampled_indices):
                sim = cosine_similarities[j_local]
                if sim < similarity_threshold:
                    weight = 1.0 - sim  # more dissimilar -> higher weight
                    graph[i][j_global] = weight
                    graph[j_global][i] = weight
                    print(f"Edge added between Node {i} and Node {j_global} | Cosine similarity: {sim:.4f}")

        print("Dissimilarity graph built successfully (using cosine similarity).")
        return graph

    def _greedy_peeling(self, graph, nodeset):
        if not nodeset: return set(), 0.0
        subgraph_nodes = set(nodeset)
        degrees = {u: sum(weight for v, weight in graph.get(u, {}).items() if v in subgraph_nodes) for u in subgraph_nodes}
        if not degrees: return set(), 0.0
        total_weight_in_subgraph = sum(degrees.values()) / 2.0
        nodes_in_subgraph = len(degrees)
        best_density = total_weight_in_subgraph / nodes_in_subgraph if nodes_in_subgraph > 0 else 0
        best_subgraph_at_peel = set(degrees.keys())
        while nodes_in_subgraph > 1:
            # This is a standard Python idiom, Pylance warning can be ignored.
            min_degree_node = min(degrees, key=degrees.get)
            node_degree = degrees[min_degree_node]
            total_weight_in_subgraph -= node_degree
            del degrees[min_degree_node]
            nodes_in_subgraph -= 1
            for neighbor, weight in graph.get(min_degree_node, {}).items():
                if neighbor in degrees:
                    degrees[neighbor] -= weight
            current_density = total_weight_in_subgraph / nodes_in_subgraph if nodes_in_subgraph > 0 else 0
            if current_density > best_density:
                best_density = current_density
                best_subgraph_at_peel = set(degrees.keys())
        return best_subgraph_at_peel, best_density

    def _calculate_density(self, graph, nodeset):
        """
        计算给定节点集的密度。
        密度定义为：(子图内所有边的权重之和) / (子图的节点数)。
        """
        if not nodeset or len(nodeset) == 0:
            return 0.0
        
        total_weight = 0.0
        nodeset_set = set(nodeset)
        for u in nodeset_set:
            for v, weight in graph[u].items():
                if v in nodeset_set:
                    total_weight += weight
        
        num_edges_or_total_weight = total_weight / 2
        num_nodes = len(nodeset_set)
        
        return num_edges_or_total_weight / num_nodes if num_nodes > 0 else 0.0
    
    def _combinatorial_dalkss(self, graph, k_min):
        """
        实现Combinatorial-DalkSS算法。
        【新策略】严格遵循论文中的描述，包括“任意添加”步骤。
        """
        print(f"最终目标数量 k_min={k_min}。采用严格遵循论文的策略...")
        
        D_current = set()
        nodes_to_consider = set(graph.keys())
        candidate_sets = []
        
        # 步骤1: 迭代构建候选集 D_j, 直到累积大小满足 k_min
        while len(D_current) < k_min and nodes_to_consider:
            H_j, _ = self._greedy_peeling(graph, nodes_to_consider)
            
            if not H_j:
                print("  警告: 在剩余节点中无法找到更多密集区域，循环终止。")
                break
            
            D_current.update(H_j)
            nodes_to_consider.difference_update(H_j)
            candidate_sets.append(D_current.copy())

        if not candidate_sets:
             print("未能生成任何候选集，算法终止。")
             return []

        print(f"共生成 {len(candidate_sets)} 个候选集。现在进行“任意添加”并评估...")

        # 步骤2 & 3: 对每个候选集 D_j 进行“任意添加”形成 D'_j 并评估
        best_final_set = set()
        max_final_density = -1.0
        all_nodes = set(graph.keys())

        for d_j in tqdm(candidate_sets, desc="评估候选集"):
            d_prime_j = d_j.copy()
            num_to_add = k_min - len(d_prime_j)

            # 如果候选集大小小于k_min, 执行“任意添加”
            if num_to_add > 0:
                nodes_available_to_add = list(all_nodes - d_prime_j)
                
                # 确保有足够的节点可供添加
                if len(nodes_available_to_add) < num_to_add:
                    print(f"警告: 可供添加的节点不足({len(nodes_available_to_add)})，无法满足 k_min={k_min}。跳过此候选集。")
                    continue
                
                # 从可用节点中随机抽样
                nodes_to_add = random.sample(nodes_available_to_add, num_to_add)
                d_prime_j.update(nodes_to_add)
            
            # 对于最后一个可能大于k_min的候选集，我们直接评估它，不进行任何操作
            # 对于之前的候选集，它们的大小现在正好是k_min
            current_density = self._calculate_density(graph, d_prime_j)

            if current_density > max_final_density:
                max_final_density = current_density
                best_final_set = d_prime_j

        if not best_final_set:
            print("警告: 未能找到任何有效的最终集合。")
            return []

        print(f"DalkSS算法完成。选出 {len(best_final_set)} 个节点，密度为 {max_final_density:.4f}。")
        return list(best_final_set)

    def select_queries(self, dataset, num_queries_to_select, similarity_threshold=1.4, num_samples_per_node=150, exclude_top_k=50):
        if not dataset or len(dataset) < num_queries_to_select:
            print("Error: Dataset is empty or smaller than the number of queries to select.")
            return [], None, None
        
        self._build_index(dataset)
        
        graph = self._build_dissimilarity_graph(
            dataset=dataset,
            similarity_threshold=similarity_threshold,
            num_samples_per_node=num_samples_per_node,
            exclude_top_k=exclude_top_k
        )

        selected_indices = self._combinatorial_dalkss(graph, k_min=num_queries_to_select)
        full_graph = graph
        if not selected_indices: return [], None, None
        
        return [dataset[i] for i in selected_indices], selected_indices, full_graph

def analyze_subgraph(selected_indices, graph):
    if not selected_indices:
        print("No nodes selected, cannot perform analysis.")
        return
    subgraph_nx = nx.Graph()
    total_weight = 0
    for i in selected_indices:
        subgraph_nx.add_node(i)
        for j, weight in graph.get(i, {}).items():
            if j in selected_indices and j > i:
                subgraph_nx.add_edge(i, j, weight=weight)
                total_weight += weight
    num_nodes = subgraph_nx.number_of_nodes()
    num_edges = subgraph_nx.number_of_edges()
    print("\n" + "="*50)
    print("               Densest Subgraph Analysis")
    print("="*50)
    print(f"  - Number of Nodes: {num_nodes}")
    print(f"  - Number of Edges: {num_edges}")
    weighted_density = total_weight / num_nodes if num_nodes > 0 else 0
    print(f"  - Weighted Density: {weighted_density:.4f}")
    unweighted_density = nx.density(subgraph_nx)
    print(f"  - Unweighted Density: {unweighted_density:.4f}")
    degrees = [d for n, d in subgraph_nx.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    print(f"  - Average Degree: {avg_degree:.2f}")
    is_connected = nx.is_connected(subgraph_nx)
    num_components = nx.number_connected_components(subgraph_nx)
    print(f"  - Is Connected: {is_connected}")
    print(f"  - Connected Components: {num_components}")
    if not is_connected:
        largest_cc = max(nx.connected_components(subgraph_nx), key=len)
        print(f"    - Size of largest connected component: {len(largest_cc)}")
    print("="*50 + "\n")

def main():
    """Main execution function"""
    try:
        with open('docs/2wikimultihopqa_backup.json', 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        print("Successfully loaded `docs/2wikimultihopqa_backup.json`.")
        input_data = all_data[:800]
    except FileNotFoundError:
        print("Warning: `docs/2wikimultihopqa_backup.json` not found. Creating dummy data for demonstration...")
        input_data = [
            {'question': 'Which NFL team has won the most Super Bowls?'},
            {'question': 'Who is the quarterback for the Kansas City Chiefs?'},
            {'question': 'What are the rules of American football?'},
            {'question': 'What is the capital of France?'},
            {'question': 'How tall is the Eiffel Tower?'},
            {'question': 'What is the main ingredient in croissant?'},
            {'question': 'Who wrote the novel "War and Peace"?'},
            {'question': 'What is the plot of Dostoevsky\'s "Crime and Punishment"?'},
            {'question': 'Tell me about the history of Russian literature.'},
            {'question': 'What is the process of photosynthesis?'},
            {'question': 'How do mitochondria generate energy?'},
            {'question': 'Describe the structure of a eukaryotic cell.'}
        ] * 20
        
    output_json_path = 'docs/2wikimultihopqa_selection_faiss.json'
    data_objects = [SimpleNamespace(**d) for d in input_data]

    num_queries_to_select = 150
    similarity_threshold = 0.249
    num_samples_per_node = 150
    exclude_top_k = 50

    # Use the local model path
    local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
    selector = L2FaissSelector(model_name=local_model_path)
    
    selected_data, selected_indices, full_graph = selector.select_queries(
        dataset=data_objects,
        num_queries_to_select=num_queries_to_select,
        similarity_threshold=similarity_threshold,
        num_samples_per_node=num_samples_per_node,
        exclude_top_k=exclude_top_k
    )
    
    if not selected_data:
        print("Error: The selection process did not return any data points. Terminating.")
        return
        
    print(f"Selection process finished. Successfully selected {len(selected_data)} data points.")

    analyze_subgraph(selected_indices, full_graph)

    selected_dicts = [vars(p) for p in selected_data]
    
    print(f"Saving {len(selected_dicts)} selected data points to {output_json_path}...")
    os.makedirs('docs', exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(selected_dicts, f, indent=4, ensure_ascii=False)
    print("Save successful!")

if __name__ == '__main__':
    main()

