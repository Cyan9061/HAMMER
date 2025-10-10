# Experiment/query_selection/graph_selector.py
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GraphSelector:
    """
    使用基于图的方法选择一个多样化的查询子集。
    该方法遵循以下步骤：
    1. 使用SentenceTransformer模型将所有查询嵌入到向量空间。
    2. 构建一个K-Farthest Neighbors (KFN)图，其中边的权重代表不相似度(1 - cosine_similarity)。
    3. 在KFN图上运行Combinatorial-DalkSS算法，找到一个大小至少为k的近似最密子图。
       这个子图的“高密度”意味着其内部节点（查询）之间具有高度的相互不相似性。
    """
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        """
        初始化选择器并加载嵌入模型。
        :param model_name: 用于生成嵌入的SentenceTransformer模型名称。
        """
        print(f"正在加载嵌入模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def _get_embeddings(self, dataset):
        """
        为数据集中的所有查询计算并归一化嵌入。
        """
        print(f"正在为 {len(dataset)} 条查询生成向量嵌入...")
        queries = [data.question for data in dataset if isinstance(data.question, str)]
        
        # 使用模型的批量处理能力以提高效率
        raw_embeddings = self.model.encode(queries, show_progress_bar=True, batch_size=32)
        
        # 归一化嵌入向量 (L2范数)
        self.embeddings = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        print("向量嵌入生成并归一化完成。")

    def _build_kfn_graph(self, k_neighbors):
        """
        构建K-Farthest Neighbors (KFN)图。
        对于每个节点，找到其k个最不相似的邻居并创建带权重的边。
        
        :param k_neighbors: 每个节点要连接的最远邻居的数量。
        :return: 图的邻接列表表示，格式为 dict[int, dict[int, float]]
                  graph[u][v] = weight, 代表u和v之间的不相似度。
        """
        if self.embeddings is None:
            raise ValueError("必须先生成向量嵌入。")

        num_nodes = self.embeddings.shape[0]
        if k_neighbors >= num_nodes:
            print("警告: k_neighbors大于或等于节点总数，将构建一个全连接图。")
            k_neighbors = num_nodes - 1

        print(f"正在构建K-Farthest Neighbors (KFN)图 (k={k_neighbors})...")
        
        # 计算所有节点对之间的余弦相似度
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # 计算不相似度 (距离) 矩阵
        dissimilarity_matrix = 1 - similarity_matrix
        
        graph = {i: {} for i in range(num_nodes)}
        
        for i in tqdm(range(num_nodes), desc="构建KFN图"):
            # 找到k个最远的邻居（不相似度最高的k个）
            # np.argpartition比np.argsort更快，如果只需要找到top k
            farthest_indices = np.argpartition(dissimilarity_matrix[i], -k_neighbors)[-k_neighbors:]
            
            for j in farthest_indices:
                if i == j: continue
                weight = dissimilarity_matrix[i, j]
                # 添加双向边
                graph[i][j] = weight
                graph[j][i] = weight
        
        print("KFN图构建完成。")
        return graph

    def _greedy_peeling(self, graph, nodeset):
        """
        Greedy Peeling算法，用于找到图G在节点集nodeset上的近似最密子图。
        这是Charikar算法的实现，提供了2-近似保证。
        
        :param graph: 图的邻接列表。
        :param nodeset: 需要在其中寻找最密子图的节点集合。
        :return: 一个元组 (best_subgraph, best_density)。
        """
        if not nodeset:
            return set(), 0.0

        # 计算初始的加权度
        degrees = {u: 0.0 for u in nodeset}
        for u in nodeset:
            for v, weight in graph[u].items():
                if v in nodeset:
                    degrees[u] += weight
        
        current_nodes = list(nodeset)
        
        best_density = 0.0
        best_subgraph = set()

        for _ in range(len(current_nodes)):
            # 计算当前子图的密度
            current_subgraph_nodes = set(degrees.keys())
            if not current_subgraph_nodes: break
            
            total_degree = sum(degrees.values())
            # 每条边被计算两次，所以总权重是 total_degree / 2
            current_density = (total_degree / 2) / len(current_subgraph_nodes)

            if current_density > best_density:
                best_density = current_density
                best_subgraph = current_subgraph_nodes.copy()

            # 找到度数最低的节点并移除
            min_degree_node = min(degrees, key=degrees.get)
            
            # 更新其邻居的度数
            for neighbor, weight in graph[min_degree_node].items():
                if neighbor in degrees:
                    degrees[neighbor] -= weight
            
            del degrees[min_degree_node]
            
        return best_subgraph, best_density

    def _combinatorial_dalkss(self, graph, k_min):
        """
        实现图片中的Combinatorial-DalkSS算法。
        
        :param graph: KFN图。
        :param k_min: 希望选出的子图的最小尺寸。
        :return: 最终选出的节点索引集合。
        """
        print(f"开始执行Combinatorial-DalkSS算法以寻找大小至少为 {k_min} 的最密子图...")
        
        D_current = set()
        nodes_to_consider = set(graph.keys())
        candidate_sets = []

        while len(D_current) < k_min and nodes_to_consider:
            # 1. 在剩余节点中找到最密子图 Hj
            Hj, _ = self._greedy_peeling(graph, nodes_to_consider)
            
            if not Hj: # 如果找不到任何子图，则退出
                break
            
            # 2. 将Hj加入当前解，并从待考虑节点中移除
            D_current.update(Hj)
            nodes_to_consider.difference_update(Hj)
            
            # 3. 保存这个中间候选解
            candidate_sets.append(D_current.copy())
        
        # 4. 从所有候选解中选择最好的一个
        # 根据论文，需要评估所有满足大小条件的候选解的密度
        print(f"共生成 {len(candidate_sets)} 个候选集，正在评估以选择最优者...")
        best_final_set = set()
        max_final_density = -1.0

        for D_j in tqdm(candidate_sets, desc="评估候选集"):
            # DalkSS算法要求评估大小至少为k的集合。
            # 我们的循环保证了最终的候选集大小会增长。
            # 我们只评估那些满足大小条件的。
            if len(D_j) >= k_min:
                _, density = self._greedy_peeling(graph, D_j)
                if density > max_final_density:
                    max_final_density = density
                    best_final_set = D_j
        
        # 如果循环结束时没有任何候选集满足大小k，则返回最后生成的那个
        if not best_final_set and candidate_sets:
            best_final_set = candidate_sets[-1]
            print(f"警告: 未找到大小>=k的候选集。返回最后生成的集合，大小为 {len(best_final_set)}")

        print(f"DalkSS算法完成。选出 {len(best_final_set)} 个节点，密度为 {max_final_density:.4f}。")
        return list(best_final_set)

    def select_queries(self, dataset, num_queries_to_select, k_neighbors_for_kfn=20):
        """
        主函数，执行整个多样性选择流程。
        
        :param dataset: 原始数据集，每个元素都是一个对象，包含 .question 属性。
        :param num_queries_to_select: 希望选出的查询数量 (k_min)。
        :param k_neighbors_for_kfn: 构建KFN图时使用的邻居数。
        :return: 选出的数据点列表。
        """
        if not dataset or len(dataset) < num_queries_to_select:
            print("错误: 数据集为空或小于要求选择的数量。")
            return []

        # 1. 计算嵌入
        self._get_embeddings(dataset)
        
        # 2. 构建KFN图
        kfn_graph = self._build_kfn_graph(k_neighbors=k_neighbors_for_kfn)
        
        # 3. 运行DalkSS算法
        selected_indices = self._combinatorial_dalkss(kfn_graph, k_min=num_queries_to_select)
        
        # 4. 映射回原始数据
        final_selected_dataset = [dataset[i] for i in selected_indices]
        
        return final_selected_dataset