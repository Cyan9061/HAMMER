"""hammer/utils/coreset.py
Complete FastCore algorithm implementation strictly following the paper:
"Efficient Coreset Selection with Cluster-based Methods"

Fixed based on detailed code review feedback.
"""

import typing as T
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

@dataclass
class CoresetResult:
    """Results of FastCore coreset selection."""
    coreset_indices: T.List[int]
    weights: T.List[float]  # 修改为float，因为归一化后不再是整数
    cluster_labels: np.ndarray
    clusters: T.List[T.List[int]]
    upper_bounds: np.ndarray

class FastCore:
    """
    Complete FastCore implementation following Algorithm 1 with optimizations.
    """
    
    def __init__(
        self,
        num_hyperplanes: int = 64,
        num_subspaces: int = 3,
        num_clusters_pq: int = 256,
        sample_size: int = 500,
        method: str = "pq",
        normalize_weights: bool = True  # 新增参数控制权重归一化
    ):
        self.num_hyperplanes = num_hyperplanes
        self.num_subspaces = num_subspaces
        self.num_clusters_pq = num_clusters_pq
        self.sample_size = sample_size
        self.method = method
        self.normalize_weights = normalize_weights  # 存储归一化选项
        
    def select_coreset(
        self,
        embeddings: np.ndarray,
        coreset_size: int,
        random_state: int = 42
    ) -> CoresetResult:
        """Main FastCore algorithm following Algorithm 1."""
        # Use local random generator for better isolation
        rng = np.random.RandomState(random_state)
        
        # Normalize embeddings for stable LSH and PQ
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        N, m = embeddings.shape
        K = coreset_size
        
        logger.info(f"🚀 FastCore: N={N}, K={K}, m={m}, method={self.method}, normalize_weights={self.normalize_weights}")
        
        # Handle corner case: coreset_size >= N
        if K >= N:
            logger.warning(f"Coreset size ({K}) >= total samples ({N})")
            # Still do LSH clustering for consistency
            clusters = self._lsh_clustering(embeddings, rng)
            weights = [1.0 / N] * N if self.normalize_weights else [1.0] * N
            return CoresetResult(
                coreset_indices=list(range(N)),
                weights=weights,
                cluster_labels=self._get_cluster_labels(N, clusters),
                clusters=clusters,
                upper_bounds=np.zeros((N, len(clusters)))
            )
        
        # ================== PREPROCESSING ==================
        logger.info("📍 Step 1: LSH Clustering")
        clusters = self._lsh_clustering(embeddings, rng)
        num_clusters = len(clusters)  # Avoid shadowing typing.T
        logger.info(f"   Created {num_clusters} clusters")
        
        logger.info("📍 Step 2: Computing upper bounds")
        if self.method == "pq":
            subspace_dim = m // self.num_subspaces
            if subspace_dim > 3:
                logger.info(f"   Using PQ (subspace_dim={subspace_dim} > 3)")
            else:
                logger.info(f"   Using PQ (subspace_dim={subspace_dim} ≤ 3, consider convex_hull)")
            upper_bounds = self._compute_upper_bounds_pq_aligned(embeddings, clusters, rng)
        else:
            # ConvexHull method with strict dimension check
            subspace_dim = m // self.num_subspaces
            if subspace_dim > 3:
                raise ValueError(f"ConvexHull method only suitable for m/M≤3, got {subspace_dim}")
            upper_bounds = self._compute_upper_bounds_convex_hull_fixed(embeddings, clusters)
        
        logger.info(f"   Upper bounds matrix: {upper_bounds.shape}")
        
        # ================== MAIN ALGORITHM ==================
        R = []
        cluster_sizes = np.array([len(c) for c in clusters], dtype=np.float32)
        current_best = np.full(num_clusters, np.inf, dtype=np.float32)
        
        logger.info("📍 Step 3: Greedy selection with O(T) evaluation")
        
        for iteration in range(K):
            logger.info(f"coreset iter {iteration + 1}/{K}")
            
            # Sample Ds ⊆ D \ R (optimized set difference)
            remaining_indices = np.setdiff1d(np.arange(N), R)
            sample_size = min(self.sample_size, len(remaining_indices))
            if sample_size == len(remaining_indices):
                Ds = remaining_indices
            else:
                Ds = rng.choice(remaining_indices, size=sample_size, replace=False)
            
            best_score = np.inf
            best_item = None
            
            # O(T) candidate evaluation with caching
            for di in Ds:
                # Vectorized score: sum of |Ct| * min(current_best[t], upper_bounds[di, t])
                cluster_mins = np.minimum(current_best, upper_bounds[di])
                score = np.dot(cluster_sizes, cluster_mins)
                
                if score < best_score:
                    best_score = score
                    best_item = di
            
            # Update coreset and cached best values
            if best_item is not None:
                R.append(best_item)
                current_best = np.minimum(current_best, upper_bounds[best_item])
                logger.info(f"     Selected item {best_item} with score {best_score:.4f}")
        
        # ================== WEIGHT COMPUTATION ==================
        logger.info("📍 Step 4: Computing weights λj")
        weights = self._compute_weights_vectorized_normalized(R, clusters, upper_bounds)
        
        logger.info(f"✅ FastCore complete! Selected {len(R)} items")
        
        return CoresetResult(
            coreset_indices=R,
            weights=weights,
            cluster_labels=self._get_cluster_labels(N, clusters),
            clusters=clusters,
            upper_bounds=upper_bounds
        )
    
    def _lsh_clustering(self, embeddings: np.ndarray, rng: np.random.RandomState) -> T.List[T.List[int]]:
        """LSH clustering following Section 5.1."""
        N, m = embeddings.shape
        p = self.num_hyperplanes
        
        # Generate p normalized random hyperplanes
        hyperplanes = rng.randn(p, m)
        hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        
        # Vectorized hashing: O(Npm)
        hash_matrix = np.dot(embeddings, hyperplanes.T) > 0
        hash_codes = [tuple(row) for row in hash_matrix]
        
        # Group into clusters
        cluster_dict = {}
        for i, code in enumerate(hash_codes):
            if code not in cluster_dict:
                cluster_dict[code] = []
            cluster_dict[code].append(i)
        
        return list(cluster_dict.values())
    
    def _compute_upper_bounds_pq_aligned(
        self,
        embeddings: np.ndarray,
        clusters: T.List[T.List[int]],
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        PQ upper bounds aligned with paper approach.
        
        Note: This implements a conservative version of the paper's PQ method
        by adding quantization error terms (ε_i + codebook_distance + ρ_t)
        to ensure strict upper bound guarantees via triangle inequality.
        The paper's original PQ uses only codebook distances as approximation.
        """
        N, m = embeddings.shape
        num_clusters = len(clusters)
        M = self.num_subspaces
        
        # Uniform subspace division with remainder handling
        subspace_sizes = np.full(M, m // M)
        subspace_sizes[:m % M] += 1
        subspace_starts = np.concatenate([[0], np.cumsum(subspace_sizes[:-1])])
        
        upper_bounds = np.zeros((N, num_clusters), dtype=np.float32)
        
        logger.info(f"   PQ: M={M} subspaces, sizes={subspace_sizes}")
        
        for ℓ in range(M):
            start_dim = subspace_starts[ℓ]
            end_dim = start_dim + subspace_sizes[ℓ]
            subspace_dim = subspace_sizes[ℓ]
            
            # Extract subspace features
            subspace_features = embeddings[:, start_dim:end_dim]
            
            # FIXED: Proper B selection without expensive unique operation
            B = min(self.num_clusters_pq, N)
            if B < 2:
                B = 2  # Minimum for KMeans
            
            if N < B:
                logger.warning(f"   Subspace {ℓ}: N({N}) < B({B}), using pairwise distances")
                # Fallback to pairwise distances
                for i in range(N):
                    for t, cluster in enumerate(clusters):
                        if cluster:
                            dists = np.linalg.norm(
                                subspace_features[i:i+1] - subspace_features[cluster], axis=1
                            )
                            upper_bounds[i, t] += dists.max()
                continue
            
            # Create codebook with proper random state
            sample_size = min(1000, N)
            sample_indices = rng.choice(N, size=sample_size, replace=False)
            kmeans = KMeans(n_clusters=B, random_state=rng.randint(2**31), n_init=10)
            kmeans.fit(subspace_features[sample_indices])
            
            # Quantize all features
            quantized_codes = kmeans.predict(subspace_features)
            centroids = kmeans.cluster_centers_
            
            # Compute quantization errors: ε_i^ℓ = ||x_i^ℓ - c_{q_i^ℓ}||
            eps_i_l = np.linalg.norm(
                subspace_features - centroids[quantized_codes], axis=1
            )
            
            # Build codebook distance matrix: ||c_i - c_j||
            codebook_distances = np.linalg.norm(
                centroids[:, None, :] - centroids[None, :, :], axis=2
            )
            
            # Cluster-level preprocessing with optimized complexity O(MNT)
            for t, cluster in enumerate(clusters):
                if not cluster:
                    continue
                
                cluster = np.array(cluster)
                cluster_codes = quantized_codes[cluster]
                cluster_eps = eps_i_l[cluster]
                
                # ρ_t^ℓ = max_{k∈Ct} ||x_k^ℓ - c_{q_k^ℓ}|| (quantization error in cluster)
                rho_t_l = cluster_eps.max()
                
                # Optimized: use unique codes in cluster to build lookup table
                unique_codes = np.unique(cluster_codes)
                cluster_table = codebook_distances[:, unique_codes].max(axis=1)
                
                # For each item i, compute conservative upper bound:
                # ||x_i^ℓ - x_k^ℓ|| ≤ ε_i^ℓ + ||c_{q_i} - c_{q_k}|| + ρ_t^ℓ
                for i in range(N):
                    q_i_l = quantized_codes[i]
                    eps_i_l_val = eps_i_l[i]
                    
                    # Three-term upper bound (conservative version of paper's PQ)
                    upper_bounds[i, t] += eps_i_l_val + cluster_table[q_i_l] + rho_t_l
        
        return upper_bounds
    
    def _compute_upper_bounds_convex_hull_fixed(
        self,
        embeddings: np.ndarray,
        clusters: T.List[T.List[int]]
    ) -> np.ndarray:
        """FIXED ConvexHull method with proper dimension checks."""
        N, m = embeddings.shape
        num_clusters = len(clusters)
        M = self.num_subspaces
        
        subspace_sizes = np.full(M, m // M)
        subspace_sizes[:m % M] += 1
        subspace_starts = np.concatenate([[0], np.cumsum(subspace_sizes[:-1])])
        
        upper_bounds = np.zeros((N, num_clusters), dtype=np.float32)
        
        for ℓ in range(M):
            start_dim = subspace_starts[ℓ]
            end_dim = start_dim + subspace_sizes[ℓ]
            subspace_dim = subspace_sizes[ℓ]
            subspace_features = embeddings[:, start_dim:end_dim]
            
            for t, cluster in enumerate(clusters):
                # FIXED: Proper threshold for convex hull (need d+1 points in d dimensions)
                if len(cluster) < subspace_dim + 1:
                    # Too few points for convex hull, use brute force
                    for i in range(N):
                        if cluster:
                            dists = np.linalg.norm(
                                subspace_features[i:i+1] - subspace_features[cluster], axis=1
                            )
                            upper_bounds[i, t] += dists.max()
                    continue
                
                try:
                    cluster_points = subspace_features[cluster]
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    
                    # Vectorized distance computation to hull vertices
                    for i in range(N):
                        dists = np.linalg.norm(
                            subspace_features[i:i+1] - hull_points, axis=1
                        )
                        upper_bounds[i, t] += dists.max()
                        
                except Exception as e:
                    logger.warning(f"ConvexHull failed for cluster {t} in subspace {ℓ}: {e}")
                    # Fallback to brute force
                    for i in range(N):
                        if cluster:
                            dists = np.linalg.norm(
                                subspace_features[i:i+1] - subspace_features[cluster], axis=1
                            )
                            upper_bounds[i, t] += dists.max()
        
        return upper_bounds
    
    def _compute_weights_vectorized_normalized(
        self,
        R: T.List[int],
        clusters: T.List[T.List[int]],
        upper_bounds: np.ndarray
    ) -> T.List[float]:
        """
        🔥 新增：归一化权重计算，确保权重总和为1
        
        Following Lines 16-17 with optional normalization:
        λj = Σ_t I[cj = argmin_{cj'∈R} u_{γ(j')t}] × |Ct|
        
        Then normalize: λj_normalized = λj / Σ_j λj
        """
        if not R:
            return []
        
        # Step 1: 计算原始权重（簇大小）
        raw_weights = [0] * len(R)
        R_array = np.array(R)
        
        # Vectorized cluster assignment
        for t, cluster in enumerate(clusters):
            if not cluster:
                continue
            
            # Find argmin_{cj ∈ R} u_{γ(j)t}
            distances = upper_bounds[R_array, t]
            assigned_idx = np.argmin(distances)
            raw_weights[assigned_idx] += len(cluster)
        
        raw_weights = np.array(raw_weights, dtype=np.float64)
        total_raw_weight = np.sum(raw_weights)
        
        logger.info(f"🔢 原始权重: {raw_weights}")
        logger.info(f"🔢 原始权重总和: {total_raw_weight}")
        
        # Step 2: 根据配置决定是否归一化
        if self.normalize_weights:
            if total_raw_weight > 0:
                normalized_weights = raw_weights / total_raw_weight
                logger.info(f"✅ 归一化权重: {normalized_weights}")
                logger.info(f"✅ 归一化权重总和: {np.sum(normalized_weights):.10f}")
                return normalized_weights.tolist()
            else:
                # 异常情况：权重总和为0，返回均匀权重
                uniform_weights = np.ones(len(R)) / len(R)
                logger.warning(f"⚠️ 权重总和为0，使用均匀权重: {uniform_weights}")
                return uniform_weights.tolist()
        else:
            # 不归一化，返回原始权重（转换为float）
            logger.info(f"ℹ️ 保持原始权重（未归一化）: {raw_weights}")
            return raw_weights.astype(np.float64).tolist()
    
    def _get_cluster_labels(self, N: int, clusters: T.List[T.List[int]]) -> np.ndarray:
        """Create cluster labels array."""
        labels = np.zeros(N, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for item_idx in cluster:
                labels[item_idx] = cluster_id
        return labels

# ================== MAIN INTERFACE ==================
def select_coreset_from_embeddings(
    embeddings: np.ndarray,
    coreset_size: int,
    random_state: int = 42,
    num_hyperplanes: int = 64,
    num_subspaces: int = 3,
    sample_size: int = 500,
    method: str = "pq",
    normalize_weights: bool = True  # 新增参数
) -> CoresetResult:
    """
    FastCore coreset selection with normalized weights option.
    
    Args:
        embeddings: Input data matrix (N, m)
        coreset_size: Number of items to select (K)
        random_state: Random seed for reproducibility
        num_hyperplanes: Number of hyperplanes for LSH clustering (p)
        num_subspaces: Number of subspaces for PQ (M)
        sample_size: Sampling size for candidate selection
        method: "pq" for Product Quantization or "convex_hull"
        normalize_weights: If True, normalize weights to sum to 1.0
    
    Returns:
        CoresetResult with selected coreset and normalized weights
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D numpy array.")
    
    fastcore = FastCore(
        num_hyperplanes=num_hyperplanes,
        num_subspaces=num_subspaces,
        sample_size=sample_size,
        method=method,
        normalize_weights=normalize_weights  # 传递归一化选项
    )
    
    return fastcore.select_coreset(embeddings, coreset_size, random_state)