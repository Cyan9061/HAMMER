"""
查询选择算法前后对比可视化脚本
显示run_selection_graph.py算法执行前后的状态对比

功能:
1. 执行前：显示所有查询的原始分布（按类型着色）
2. 执行后：高亮选中的查询，标灰未选中的查询

作者: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.manifold import TSNE
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore')

# 导入基本库
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_qa_data(file_path, max_samples=1000):
    """
    从JSON文件加载QA数据
    
    参数:
        file_path (str): JSON文件路径
        max_samples (int): 最大加载样本数
    
    返回:
        list: QA数据列表
    """
    print(f"📁 正在加载QA数据: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    qa_data = []
    for i, line in enumerate(lines[:max_samples]):
        if line.strip():
            try:
                data = json.loads(line.strip())
                qa_data.append({
                    'id': data.get('id', f'qa_{i}'),
                    'query': data['query'],
                    'answer': data['answer_ground_truth'],
                    'type': data.get('metadata', {}).get('type', 'unknown'),
                    'difficulty': data.get('metadata', {}).get('difficulty', 'hard')
                })
            except json.JSONDecodeError:
                print(f"⚠️  警告: 无法解析第 {i+1} 行")
                continue
    
    print(f"✅ 成功加载 {len(qa_data)} 个QA对")
    return qa_data

def reduce_to_2d_tsne(embeddings):
    """
    使用t-SNE降维到2D
    
    参数:
        embeddings (numpy.ndarray): 高维嵌入向量
    
    返回:
        numpy.ndarray: 2D坐标
    """
    print(f"📊 使用t-SNE降维到2D...")
    
    perplexity = min(30, max(5, len(embeddings) // 4))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"✅ 降维完成: {embeddings_2d.shape}")
    return embeddings_2d

def create_before_visualization(qa_data, embeddings_2d, cluster_labels, save_path):
    """
    创建算法执行前的可视化图（全部用deep颜色+黑色边框+尺寸200）
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标
        cluster_labels (numpy.ndarray): 聚类标签
        save_path (str): 保存路径
    """
    print("🎨 创建执行前的可视化图...")
    
    plt.figure(figsize=(10, 8))
    
    # 使用deep颜色方案和basic标记方案
    colors_deep = [
        '#000080',  # 海军蓝
        '#FF8C00',  # 深橙
        '#006400',  # 深绿
        '#4B0082',  # 靛青
        '#8B4513',  # 鞍褐
        '#2F4F4F',  # 深青灰
        '#800080',  # 紫色
        '#556B2F',  # 深橄榄绿
        '#008B8B',  # 深青
        '#B8860B'   # 深金黄
    ]
    
    markers_basic = ['o', 's', '^', 'v', 'D', '*', 'p', 'h', '+', 'x']
    
    unique_clusters = np.unique(cluster_labels)
    
    # 为每个聚类绘制不同颜色和形状的点，全部用原始颜色+黑色边框
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        color = colors_deep[i % len(colors_deep)]
        marker = markers_basic[i % len(markers_basic)]
        
        plt.scatter(
            embeddings_2d[cluster_mask, 0],
            embeddings_2d[cluster_mask, 1],
            c=color,
            marker=marker,
            s=200,                    # 更大尺寸
            alpha=0.9,               # 高透明度
            edgecolors='black',      # 黑色边框
            linewidth=1.5,           # 边框粗细
            zorder=5
        )
    
    # 设置标题
    # plt.title('Deep Colors + Basic Markers - Original Dataset', 
    #           fontsize=16, pad=20)
    
    # 设置坐标轴
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, alpha=0.4)
    
    # 设置坐标轴范围
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1024, bbox_inches='tight', facecolor='white')
    print(f"💾 原始图已保存: {save_path}")
    
    return plt.gcf()

def identify_points_to_remove(embeddings_2d, cluster_labels, edge_deletion_percentage=0.6, random_deletion_percentage=0.9):
    """
    分两轮删除识别每个聚类中的点：第一轮删除最边缘的点，第二轮在剩余点中随机删除点
    
    参数:
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        cluster_labels (numpy.ndarray): 聚类标签数组
        edge_deletion_percentage (float): 第一轮删除最边缘点的百分比，默认0.6（60%）
        random_deletion_percentage (float): 第二轮在剩余点中随机删除点的百分比，默认0.5（50%）
    
    返回:
        numpy.ndarray: 布尔数组，True表示该点需要被删除
    """
    print(f"🎯 正在执行两轮删除...")
    print(f"   第一轮(删除最边缘): {edge_deletion_percentage*100:.0f}%")
    print(f"   第二轮(剩余点中随机删除): {random_deletion_percentage*100:.0f}%")
    
    unique_clusters = np.unique(cluster_labels)
    deletion_mask = np.zeros(len(embeddings_2d), dtype=bool)
    
    total_original = 0
    total_after_round1 = 0
    total_after_round2 = 0
    
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的所有点
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) <= 1:
            continue  # 跳过只有一个点的聚类
        
        original_count = len(cluster_points)
        total_original += original_count
        
        # 计算聚类中心（质心）
        cluster_center = np.mean(cluster_points, axis=0)
        
        # 计算每个点到聚类中心的距离
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        
        # 第一轮：删除最边缘的点
        n_edge_deletions = int(len(cluster_points) * edge_deletion_percentage)
        edge_deletion_indices = []
        
        if n_edge_deletions > 0:
            # 找到距离最远的n_edge_deletions个点
            edge_deletion_indices = np.argsort(distances)[-n_edge_deletions:]
        
        # 第一轮删除后剩余的点
        remaining_indices = [i for i in range(len(cluster_points)) 
                           if i not in edge_deletion_indices]
        remaining_count = len(remaining_indices)
        total_after_round1 += remaining_count
        
        print(f"   聚类{cluster_id}: 原始{original_count}个点 → 第一轮删除{n_edge_deletions}个边缘点 → 剩余{remaining_count}个点")
        
        # 第二轮：在剩余的点中随机删除
        n_random_deletions = int(remaining_count * random_deletion_percentage)
        random_deletion_indices = []
        
        if n_random_deletions > 0 and remaining_count > 0:
            # 在剩余点中随机选择删除点
            np.random.seed(42 + cluster_id)  # 设置随机种子以确保结果可复现
            random_selected = np.random.choice(
                remaining_indices, 
                size=min(n_random_deletions, remaining_count), 
                replace=False
            )
            random_deletion_indices = list(random_selected)
        
        final_remaining = remaining_count - len(random_deletion_indices)
        total_after_round2 += final_remaining
        
        print(f"   聚类{cluster_id}: 剩余{remaining_count}个点 → 第二轮随机删除{len(random_deletion_indices)}个点 → 最终保留{final_remaining}个点")
        
        # 合并所有要删除的点的索引
        all_deletion_indices = list(edge_deletion_indices) + random_deletion_indices
        
        # 转换为在原始数组中的索引并标记删除
        if all_deletion_indices:
            deletion_indices_global = cluster_indices[all_deletion_indices]
            deletion_mask[deletion_indices_global] = True
    
    deletion_count = np.sum(deletion_mask)
    total_count = len(embeddings_2d)
    
    print(f"\n📊 两轮删除总结:")
    print(f"   原始总点数: {total_original}")
    print(f"   第一轮后剩余: {total_after_round1} ({total_after_round1/total_original*100:.1f}%)")
    print(f"   第二轮后剩余: {total_after_round2} ({total_after_round2/total_original*100:.1f}%)")
    print(f"   总删除点数: {deletion_count} ({deletion_count/total_count*100:.1f}%)")
    
    return deletion_mask

def create_after_visualization(qa_data, embeddings_2d, cluster_labels, deletion_mask, save_path):
    """
    创建删除点后的可视化图（保留点用尺寸300，删除点标灰色）
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标
        cluster_labels (numpy.ndarray): 聚类标签
        deletion_mask (numpy.ndarray): 删除掩码，True表示删除的点
        save_path (str): 保存路径
    """
    print("🎨 创建删除点后的可视化图...")
    
    plt.figure(figsize=(10, 8))
    
    # 使用deep颜色方案和basic标记方案
    colors_deep = [
        '#000080',  # 海军蓝
        '#FF8C00',  # 深橙
        '#006400',  # 深绿
        '#4B0082',  # 靛青
        '#8B4513',  # 鞍褐
        '#2F4F4F',  # 深青灰
        '#800080',  # 紫色
        '#556B2F',  # 深橄榄绿
        '#008B8B',  # 深青
        '#B8860B'   # 深金黄
    ]
    
    markers_basic = ['o', 's', '^', 'v', 'D', '*', 'p', 'h', '+', 'x']
    
    unique_clusters = np.unique(cluster_labels)
    
    # 为每个聚类绘制点，区分删除和保留
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        color = colors_deep[i % len(colors_deep)]
        marker = markers_basic[i % len(markers_basic)]
        
        # 分离删除和保留的点
        deleted_in_cluster = [idx for idx in cluster_indices if deletion_mask[idx]]
        kept_in_cluster = [idx for idx in cluster_indices if not deletion_mask[idx]]
        
        # 绘制删除的点（灰色，较小尺寸）
        if deleted_in_cluster:
            plt.scatter(
                embeddings_2d[deleted_in_cluster, 0],
                embeddings_2d[deleted_in_cluster, 1],
                c='lightgray',
                marker=marker,
                s=120,
                alpha=0.4,
                zorder=1
            )
        
        # 绘制保留的点（原始颜色，大尺寸）
        if kept_in_cluster:
            plt.scatter(
                embeddings_2d[kept_in_cluster, 0],
                embeddings_2d[kept_in_cluster, 1],
                c=color,
                marker=marker,
                s=300,                   # 大尺寸
                alpha=0.9,
                edgecolors='black',
                linewidth=1.5,
                zorder=5
            )
    
    # 设置标题
    deleted_count = np.sum(deletion_mask)
    kept_count = len(embeddings_2d) - deleted_count
    # plt.title(f'Deep Colors + Basic Markers - After Filtering\n{kept_count} Kept, {deleted_count} Removed', 
    #           fontsize=16, pad=20)
    
    # 设置坐标轴
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, alpha=0.4)
    
    # 设置坐标轴范围
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1024, bbox_inches='tight', facecolor='white')
    print(f"💾 筛选后图已保存: {save_path}")
    print(f"📊 显示了 {kept_count} 个保留的点，{deleted_count} 个删除的点")
    
    return plt.gcf()

def analyze_selection_strategy(qa_data, selected_indices, embeddings_2d):
    """
    分析查询选择策略的效果
    
    参数:
        qa_data (list): QA数据列表
        selected_indices (list): 选中的查询索引
        embeddings_2d (numpy.ndarray): 2D坐标
    """
    print("\n" + "="*70)
    print("                    🔍 查询选择策略分析")
    print("="*70)
    
    # 1. 基本统计
    total_queries = len(qa_data)
    selected_queries = len(selected_indices)
    selection_rate = (selected_queries / total_queries) * 100
    
    print(f"📊 基本统计:")
    print(f"   总查询数: {total_queries}")
    print(f"   选中查询数: {selected_queries}")
    print(f"   选择率: {selection_rate:.2f}%")
    
    # 2. 类型分布对比
    print(f"\n📋 类型分布对比:")
    
    # 原始分布
    original_types = {}
    for qa in qa_data:
        qa_type = qa['type']
        original_types[qa_type] = original_types.get(qa_type, 0) + 1
    
    # 选中分布
    selected_types = {}
    for idx in selected_indices:
        qa_type = qa_data[idx]['type']
        selected_types[qa_type] = selected_types.get(qa_type, 0) + 1
    
    print(f"   {'类型':<20} {'原始':<10} {'选中':<10} {'选择率':<10}")
    print(f"   {'-'*50}")
    for qa_type in sorted(original_types.keys()):
        original_count = original_types[qa_type]
        selected_count = selected_types.get(qa_type, 0)
        type_selection_rate = (selected_count / original_count) * 100
        print(f"   {qa_type:<20} {original_count:<10} {selected_count:<10} {type_selection_rate:<10.1f}%")
    
    # 3. 空间分布分析
    print(f"\n🗺️  空间分布分析:")
    selected_coords = embeddings_2d[selected_indices]
    all_coords = embeddings_2d
    
    # 计算分散程度（标准差）
    selected_std = np.std(selected_coords, axis=0)
    all_std = np.std(all_coords, axis=0)
    
    print(f"   选中查询空间分散度: X={selected_std[0]:.3f}, Y={selected_std[1]:.3f}")
    print(f"   所有查询空间分散度: X={all_std[0]:.3f}, Y={all_std[1]:.3f}")
    print(f"   相对分散度: {np.mean(selected_std/all_std):.3f}")
    
    # 4. 距离分析
    print(f"\n📏 距离分析:")
    
    # 计算选中查询之间的平均距离
    if len(selected_indices) > 1:
        from scipy.spatial.distance import pdist
        selected_distances = pdist(selected_coords)
        avg_selected_distance = np.mean(selected_distances)
        
        # 随机抽样相同数量的查询作为对比
        np.random.seed(42)
        random_indices = np.random.choice(len(qa_data), len(selected_indices), replace=False)
        random_coords = embeddings_2d[random_indices]
        random_distances = pdist(random_coords)
        avg_random_distance = np.mean(random_distances)
        
        print(f"   选中查询间平均距离: {avg_selected_distance:.3f}")
        print(f"   随机选择间平均距离: {avg_random_distance:.3f}")
        print(f"   距离比率: {avg_selected_distance/avg_random_distance:.3f}")
        
        if avg_selected_distance > avg_random_distance:
            print("   ✅ 算法成功选择了更加分散的查询（提高多样性）")
        else:
            print("   ⚠️  算法选择的查询聚集程度高于随机选择")
    
    print("="*70)

def perform_clustering(embeddings_2d, n_clusters=8):
    """对2D嵌入向量执行K-means聚类分析"""
    if len(embeddings_2d) < n_clusters:
        n_clusters = len(embeddings_2d)
    
    print(f"🔍 正在执行K-means聚类，聚类数: {n_clusters}...")
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=250
    )
    
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    print(f"✅ 聚类完成，发现 {len(np.unique(cluster_labels))} 个聚类")
    return cluster_labels

def main():
    """
    主函数：生成原始图和筛选对比可视化（deep+basic风格）
    """
    print("🚀 开始生成原始图和筛选对比可视化...")
    print("=" * 70)
    
    # 1. 设置参数
    max_samples = 1000
    edge_deletion_percentage = 0.5  # 第一轮删除60%的边缘点
    random_deletion_percentage = 0.85  # 第二轮在剩余点中删除50%
    
    # 2. 找到数据文件
    data_file = 'docs/dataset/unified/2wikimultihopqa_qa_unified.json'
    if not os.path.exists(data_file):
        alternative_paths = [
            '../../docs/dataset/unified/2wikimultihopqa_qa_unified.json',
            '../../docs/dataset/unified/2wikimultihopqa_qa_unified.json'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                data_file = path
                break
        else:
            print(f"❌ 错误: 找不到数据文件")
            return
    
    # 3. 设置模型
    try:
        local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if os.path.exists(local_model_path):
            model_name = local_model_path
            print(f"🔍 使用本地模型: {model_name}")
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
            print(f"🌐 使用在线模型: {model_name}")
    except:
        model_name = 'BAAI/bge-large-en-v1.5'
        print(f"🌐 使用在线模型: {model_name}")
    
    # 4. 加载数据
    qa_data = load_qa_data(data_file, max_samples=max_samples)
    if not qa_data:
        print("❌ 没有加载到数据")
        return
    
    # 5. 生成嵌入向量
    print(f"🤖 正在加载嵌入模型: {model_name}...")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    model = HuggingFaceEmbedding(model_name=model_name, device='cuda:2')
    
    print(f"🔄 正在为 {len(qa_data)} 个问题生成嵌入向量...")
    texts = [qa['query'] for qa in qa_data]
    embeddings_list = model.get_text_embedding_batch(texts, show_progress=True)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"✅ 生成嵌入完成，形状: {embeddings.shape}")
    
    # 6. 降维到2D
    embeddings_2d = reduce_to_2d_tsne(embeddings)
    
    # 7. 执行聚类分析
    cluster_labels = perform_clustering(embeddings_2d, n_clusters=8)
    
    # 8. 识别需要删除的点
    deletion_mask = identify_points_to_remove(
        embeddings_2d, 
        cluster_labels, 
        edge_deletion_percentage=edge_deletion_percentage,
        random_deletion_percentage=random_deletion_percentage
    )
    
    # 9. 创建输出目录
    output_dir = 'visualizations/sele/dark'
    os.makedirs(output_dir, exist_ok=True)
    
    # 10. 生成对比可视化（PDF格式）
    before_path = f'{output_dir}/original_deep_basic_{max_samples}.pdf'
    after_path = f'{output_dir}/filtered_deep_basic_{max_samples}.pdf'
    
    # 创建原始图
    create_before_visualization(qa_data, embeddings_2d, cluster_labels, before_path)
    
    # 创建筛选后的图
    create_after_visualization(qa_data, embeddings_2d, cluster_labels, deletion_mask, after_path)
    
    # 11. 分析筛选结果
    analyze_filtering_results(qa_data, deletion_mask, embeddings_2d, cluster_labels)
    
    print("=" * 70)
    print("🎉 可视化完成！")
    print(f"📍 原始图: {os.path.abspath(before_path)}")
    print(f"📍 筛选图: {os.path.abspath(after_path)}")
    
    deleted_count = np.sum(deletion_mask)
    kept_count = len(embeddings_2d) - deleted_count
    print(f"🎯 筛选结果: 保留 {kept_count} 个点，删除 {deleted_count} 个点")
    print(f"🎨 使用 deep 颜色方案 + basic 标记方案")
    print(f"📁 文件保存在: {os.path.abspath(output_dir)}")

def analyze_filtering_results(qa_data, deletion_mask, embeddings_2d, cluster_labels):
    """
    分析筛选结果的效果
    
    参数:
        qa_data (list): QA数据列表
        deletion_mask (numpy.ndarray): 删除掩码
        embeddings_2d (numpy.ndarray): 2D坐标
        cluster_labels (numpy.ndarray): 聚类标签
    """
    print("\n" + "="*70)
    print("                    🔍 筛选结果分析")
    print("="*70)
    
    # 1. 基本统计
    total_points = len(qa_data)
    deleted_points = np.sum(deletion_mask)
    kept_points = total_points - deleted_points
    deletion_rate = (deleted_points / total_points) * 100
    
    print(f"📊 基本统计:")
    print(f"   总点数: {total_points}")
    print(f"   删除点数: {deleted_points}")
    print(f"   保留点数: {kept_points}")
    print(f"   删除率: {deletion_rate:.2f}%")
    
    # 2. 各聚类的删除情况
    print(f"\n📋 各聚类删除情况:")
    unique_clusters = np.unique(cluster_labels)
    
    print(f"   {'聚类ID':<8} {'总数':<8} {'删除':<8} {'保留':<8} {'删除率':<10}")
    print(f"   {'-'*50}")
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        total_in_cluster = len(cluster_indices)
        deleted_in_cluster = np.sum(deletion_mask[cluster_indices])
        kept_in_cluster = total_in_cluster - deleted_in_cluster
        cluster_deletion_rate = (deleted_in_cluster / total_in_cluster) * 100
        
        print(f"   {cluster_id:<8} {total_in_cluster:<8} {deleted_in_cluster:<8} {kept_in_cluster:<8} {cluster_deletion_rate:<10.1f}%")
    
    # 3. 空间分布分析
    print(f"\n🗺️  空间分布分析:")
    kept_coords = embeddings_2d[~deletion_mask]
    deleted_coords = embeddings_2d[deletion_mask]
    
    if len(kept_coords) > 0 and len(deleted_coords) > 0:
        # 计算分散程度（标准差）
        kept_std = np.std(kept_coords, axis=0)
        deleted_std = np.std(deleted_coords, axis=0)
        all_std = np.std(embeddings_2d, axis=0)
        
        print(f"   保留点空间分散度: X={kept_std[0]:.3f}, Y={kept_std[1]:.3f}")
        print(f"   删除点空间分散度: X={deleted_std[0]:.3f}, Y={deleted_std[1]:.3f}")
        print(f"   总体空间分散度: X={all_std[0]:.3f}, Y={all_std[1]:.3f}")
        
        # 计算平均距离到中心
        center = np.mean(embeddings_2d, axis=0)
        kept_distances = np.linalg.norm(kept_coords - center, axis=1)
        deleted_distances = np.linalg.norm(deleted_coords - center, axis=1)
        
        print(f"   保留点到中心平均距离: {np.mean(kept_distances):.3f}")
        print(f"   删除点到中心平均距离: {np.mean(deleted_distances):.3f}")
        
        if np.mean(deleted_distances) > np.mean(kept_distances):
            print("   ✅ 删除的点更靠近边缘（符合预期）")
        else:
            print("   ⚠️  删除的点更靠近中心")
    
    print("="*70)

if __name__ == '__main__':
    main()