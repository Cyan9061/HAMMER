import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import pdb
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class SubmodularSelector:
    def __init__(self, model=None, lamda=0.5):
        if model is None:
            model = os.getenv("BGE_LARGE_MODEL_PATH", "")
        self.model = SentenceTransformer(model) if model else None
        self.lamda = lamda
        self.all_queries_embeddings = None
        self.avg_original_embedding = None
        self.selected_queries = []
    
    def _compute_embeddings(self, queries):
        print(f"Generating embeddings for {len(queries)} queries...")
        
        embeddings = []
        for query in tqdm(queries, desc="Computing embeddings"):
            # 使用模型生成查询的嵌入
            embedding = self.model.encode(query)
            # 归一化嵌入向量
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return embeddings
    def diversity_function(self, current_selection_embeddings, candidate_embedding=None):
        """
        计算多样性函数的边际增益。
        这里的f(S)我们将设计为：S中每个元素到S中其他所有元素的最小距离之和。
        或者，更简单地，考虑max-min的边际贡献。
        为了配合贪心算法，我们计算添加一个新元素后，多样性分数的增益。
        这里的f(S)可以定义为：在S中，所有元素到S中所有其他元素的最小距离之和。
        或者，更贴合Max-Min，定义为：当前已选集合S中所有元素与未选元素的最大最小距离之和。
        这里我们简化一下，定义为：在S中所有元素到S中所有其他元素的最小距离的平均值。
        
        更直接地，我们沿用Max-Min的思路：新加入的元素，它与已选集合中距离最近的元素的距离，
        在整个选定过程中，我们希望这个“最近距离”最大化。
        
        为了使其成为子模函数的计算方式，我们可以考虑：
        f(S) = sum_{q_i in S} min_{q_j in S, j!=i} (distance(q_i, q_j)) # 这种不是单调的，不适合
        
        我们使用一个更适合作为子模函数多样性度量的：
        F(S) = sum_{q_i in V} max_{q_j in S} (similarity(q_i, q_j)) # 设施选址型
        或者，另一种定义：
        F(S) = sum_{q_i in S} sum_{q_j in V \ S} (distance(q_i, q_j))
        
        这里我们采用最常见的子模多样性函数之一：
        F(S) = sum_{q in V} max_{s in S} sim(q, s) (Max-Coverage / Facility Location)
        
        因此，边际增益为：
        f(S U {x}) - f(S) = sum_{q in V} (max(max_{s in S} sim(q,s), sim(q,x)) - max_{s in S} sim(q,s))
        = sum_{q in V where sim(q,x) > max_{s in S} sim(q,s)} (sim(q,x) - max_{s in S} sim(q,s))
        这个计算需要遍历所有queries，在每次迭代中可能效率不高。
        
        我们简化一下，为了更好地和你的Max-Min代码结合，我们依然使用Max-Min的思路，
        但将其改造为子模函数的边际收益：
        假设 f(S) 是已选集合 S 中所有元素对之间的最小距离之和。
        f(S U {x}) - f(S) 
        我们这里定义 f(S) 为“已选集合S中，每个元素到S中距离它最近的元素的距离之和”，
        或者更简单地，f(S)为Max-Min的选点逻辑下，被选中的点的累计最小距离。
        
        让我们重新定义 f(S) 为已选集合S的“最大最小距离和”，即 Max-Min 过程中累积的最小距离。
        这样，f(S U {x}) 的边际增益就是 `min_distance_to_selected`。
        这个边际收益的定义更贴合你的原始代码逻辑，且是递减的。
        """
        # 为了简洁和匹配Max-Min的启发式，我们直接使用max_min_distance作为边际收益的近似
        # 在子模函数理论中，这通常是用于选择下一个点的启发式，而不是直接的F(S)定义。
        # 严格的子模函数F(S)例如：F(S) = sum_{q in V} max_{s in S} (1 - dist(q, s))
        
        # 假设 current_min_distances 存储了每个未选择查询到当前 selected_embeddings 中最近查询的距离
        # 当我们添加一个新元素 x 时，我们需要更新这个列表
        # 这里，我们考虑的是，如果选择 candidate_embedding，它会带来的“多样性增益”
        # 也就是它到已选集合的最小距离。
        
        if not current_selection_embeddings.shape[0]: # 如果是第一个点
            return 0 # 第一个点的多样性边际收益为0，或根据实际定义为1（基准点）
        
        # 计算 candidate_embedding 到 current_selection_embeddings 中每个点的余弦相似度
        similarities_to_selected = cosine_similarity(
            [candidate_embedding], 
            current_selection_embeddings
        )[0]
        
        # 将相似度转换为距离 (1 - similarity)
        distances_to_selected = 1 - similarities_to_selected
        
        # 选择最小距离（Max-Min策略的核心）
        min_distance_to_selected = np.min(distances_to_selected)
        
        # 在这里，我们把多样性的边际收益定义为新加入元素到已选集合的最小距离。
        # 这个定义在贪心算法中可以work，因为我们总是选择能最大化这个“最小距离”的元素。
        # 对于F(S) = sum_{q in S} min_dist(q, S\{q}) 这种形式，很难直接计算边际增益。
        # 我们这里用一个更符合直觉的F(S)来近似：
        # F_diversity(S) = sum_{i in S} (min_j in S, j!=i distance(q_i, q_j)) // 这种不是单调的
        
        # 让我们采用 Facility Location 作为多样性子模函数 F(S) = sum_{q in V} max_{s in S} sim(q, s)
        # 或者更常见的：F(S) = sum_{q in V} max_{s in S} (1 - dist(q, s))
        # 它的边际增益是： sum_{q in V} (max(max_{s in S} (1-dist(q,s)), (1-dist(q,x))) - max_{s in S} (1-dist(q,s)))
        # = sum_{q in V} (max(current_max_sim[q], new_sim_to_x[q]) - current_max_sim[q])
        # 这里需要维护一个所有未选点到已选集合的最大相似度，以便高效计算。

        # 为了简化并与你的Max-Min代码贴近，我们假设f的边际增益就是该点到已选集合的最小距离
        # (即Max-Min选择的那个值)。在实际子模函数中，这只是一个贪心选择的启发式。
        # 我们可以定义 f(S) 为已选集合S内所有点之间的平均距离
        # f(S) = sum_{s1, s2 in S, s1!=s2} dist(s1, s2) / |S|*(|S|-1)/2
        # 这样计算边际增益非常复杂，且不适合直接放入循环。
        
        # 考虑到代码效率和你的Max-Min策略，我们直接让f的边际增益是该点到已选集合的最小距离。
        # 虽然这不严格符合某个标准的子模函数F(S)的边际增益定义，但在贪心算法中可以工作。
        return min_distance_to_selected

    def similarity_function(self, current_selection_embeddings, candidate_embedding=None):
        """
        计算相似度函数的边际增益。
        g(S) 旨在衡量子集S与原始数据集的相似度。
        我们可以将 g(S) 定义为子集S的平均嵌入向量与原始数据集的平均嵌入向量之间的余弦相似度。
        
        g(S) = cosine_similarity(mean(embedding(S)), mean(embedding(V)))
        
        当我们添加一个元素 x 到 S 时，
        g(S U {x}) - g(S) = cosine_similarity(mean(embedding(S U {x})), mean(embedding(V)))
                            - cosine_similarity(mean(embedding(S)), mean(embedding(V)))
        
        这需要每次迭代重新计算平均嵌入，但这比遍历所有V的边际增益计算要快。
        """
        # 计算当前集合的平均嵌入
        current_mean_embedding = np.mean(current_selection_embeddings, axis=0)
        
        # 计算加入候选后新集合的平均嵌入
        if candidate_embedding is not None:
            # 假设 candidate_embedding 是要加入的下一个
            new_selection_embeddings = np.vstack([current_selection_embeddings, candidate_embedding])
            new_mean_embedding = np.mean(new_selection_embeddings, axis=0)
        else: # 仅计算当前集合的相似度，用于基准
            new_mean_embedding = current_mean_embedding

        # 确保嵌入向量归一化
        current_mean_embedding_norm = current_mean_embedding / np.linalg.norm(current_mean_embedding)
        new_mean_embedding_norm = new_mean_embedding / np.linalg.norm(new_mean_embedding)
        
        # 计算与原始数据集平均嵌入的余弦相似度
        sim_current = cosine_similarity([current_mean_embedding_norm], [self.avg_original_embedding])[0][0]
        sim_new = cosine_similarity([new_mean_embedding_norm], [self.avg_original_embedding])[0][0]
        
        return sim_new - sim_current # 边际增益
    def select_optimized_queries(self, dataset, num_queries_to_select=20):
        """
        使用子模函数最大化（结合多样性和相似度）选择查询。
        :param dataset: 原始问答数据集，每个元素应包含 'question' 字段。
        :param num_queries_to_select: 希望抽取的验证集大小。
        :return: 抽取的验证集数据点列表。
        """
        if not dataset or len(dataset) == 0:
            return []
        print(dataset[0])
        pdb.set_trace()
        queries = [data_point.question for data_point in dataset if isinstance(data_point.question, str)]

        if len(queries) < num_queries_to_select:
            print(f"Warning: Number of queries ({len(queries)}) is less than desired queries to select ({num_queries_to_select}). Adjusting num_queries_to_select to {len(queries)}.")
            num_queries_to_select = len(queries)
        pdb.set_trace()
        self.all_queries_embeddings = self._compute_embeddings(queries)
        self.avg_original_embedding = np.mean(self.all_queries_embeddings, axis=0)
        self.avg_original_embedding = self.avg_original_embedding / np.linalg.norm(self.avg_original_embedding) # 归一化

        selected_indices = []
        selected_embeddings = np.array([]) # 存储已选的嵌入，用于高效计算
        
        # 贪心算法
        # pdb.set_trace()
        for i,data in enumerate(tqdm(range(num_queries_to_select), desc="Selecting queries using submodular optimization")):
            best_candidate_idx = -1
            max_marginal_gain = -float('inf')
            # pdb.set_trace()
            # 如果是第一次迭代，随机选择一个起点（或者选择一个具有最大多样性增益的初始点）
            if data == 0:
                # 可以选择随机点，或者选择与所有其他点平均距离最大的点作为起点
                # 为了简化，我们依然选择一个随机点作为起点，这是子模最大化中常见的做法。
                # 但如果你的多样性函数f(S)定义为sum_{q in V} max_{s in S} sim(q,s)，那么第一个点
                # 的选择将是与所有V中最相似的那个点。
                # 考虑到你的Max-Min起点是随机的，这里保持一致。
                first_idx = np.random.choice(len(queries))
                selected_indices.append(first_idx)
                selected_embeddings = np.expand_dims(self.all_queries_embeddings[first_idx], axis=0)
                continue # 进入下一轮迭代，开始正式选择
            
            # 遍历所有未选择的候选元素
            for candidate_idx in range(len(queries)):
                if candidate_idx in selected_indices:
                    continue

                candidate_embedding = self.all_queries_embeddings[candidate_idx]
                
                # 计算多样性函数的边际增益 f(S U {x}) - f(S)
                # 这里我们沿用Max-Min的思路：新加入的元素，它与已选集合中距离最近的元素的距离。
                # 这边际收益 f_gain 越大，多样性越高。
                f_gain = self.diversity_function(selected_embeddings,candidate_embedding) # 使用最小距离作为多样性的边际增益

                # 计算相似度函数的边际增益 g(S U {x}) - g(S)
                g_gain = self.similarity_function(selected_embeddings, candidate_embedding)
                
                # 计算总的边际增益
                current_marginal_gain = self.lamda * f_gain + (1-self.lamda) * g_gain

                if current_marginal_gain > max_marginal_gain:
                    max_marginal_gain = current_marginal_gain
                    best_candidate_idx = candidate_idx
            
            if best_candidate_idx != -1:
                selected_indices.append(best_candidate_idx)
                # 将新的嵌入添加到已选集合中
                if selected_embeddings.shape[0] == 0:
                    selected_embeddings = np.expand_dims(self.all_queries_embeddings[best_candidate_idx], axis=0)
                else:
                    selected_embeddings = np.vstack([selected_embeddings, self.all_queries_embeddings[best_candidate_idx]])
            else:
                break 
        
        print(f"Selected {len(selected_indices)} queries using submodular optimization.")
        # pdb.set_trace()
        final_selected_dataset = [dataset[idx] for idx in selected_indices]
        # print(f"Final selected dataset: {(final_selected_dataset)}")
        return final_selected_dataset

    def select_diverse_queries(self, dataset, num_diverse_queries=10, model_path=os.getenv("DEFAULT_EMBEDDING_MODEL", "")):

        if not dataset or len(dataset) == 0:
            return []

        print(f"Loading Sentence Transformer model: {model_path}...")
        queries = [data_point.question for data_point in dataset if isinstance(data_point.question, str)]
        print(f"Generating embeddings for {len(queries)} queries...")

        query_embeddings = [self.model.encode(query) for query in tqdm(queries, desc="Computing embeddings")]

        print("\n--- Strategy: based on distance ---")
        if len(queries) < num_diverse_queries:
            print(f"Warning: Number of queries ({len(queries)}) is less than desired diverse queries ({num_diverse_queries}). Adjusting num_diverse_queries to {len(queries)}.")
            num_diverse_queries = len(queries)
        
        selected_indices = []
        selected_queries = []

        first_idx = np.random.choice(len(queries))
        selected_indices.append(first_idx)
        selected_queries.append(queries[first_idx])

        for _ in tqdm(range(num_diverse_queries - 1), desc="Selecting max-min diverse queries"):
            max_min_distance = -1
            best_candidate_idx = -1

            for candidate_idx in range(len(queries)):
                if candidate_idx in selected_indices:
                    continue

                candidate_embedding = query_embeddings[candidate_idx]
                
                distances_to_selected = [
                    1 - cosine_similarity([candidate_embedding], [query_embeddings[s_idx]])[0][0]
                    for s_idx in selected_indices
                ]
                
                min_distance_to_selected = min(distances_to_selected)

                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    best_candidate_idx = candidate_idx
            
            if best_candidate_idx != -1:
                selected_indices.append(best_candidate_idx)
                selected_queries.append(queries[best_candidate_idx])
            else:
                break
                
        print(f"Selected {len(selected_queries)} max-min diverse queries.")
        print("Max-Min Diverse Queries:", selected_queries)

        return [data_point for i, data_point in enumerate(dataset) if i in selected_indices]
        
