"""
QA聚类可视化脚本
专门用于生成2wikimultihopqa数据集中QA对的语义聚类二维可视化图

作者: Claude Code Assistant
功能: 读取QA数据 -> 生成嵌入 -> t-SNE降维 -> K-means聚类 -> 按聚类着色可视化
输出: qa_embeddings_2d_clusters_1000.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import warnings
warnings.filterwarnings('ignore')

# ==================== 可配置参数 ====================
# 🎨 字体设置
FONT_FAMILY = 'Linux Libertine'  # 主字体
FONT_SIZE_TITLE = 16            # 标题字体大小
FONT_SIZE_LABEL = 30            # 坐标轴标签字体大小
FONT_SIZE_TICK = 30             # 坐标轴刻度字体大小

# 📏 图片尺寸设置 (英寸)
FIGURE_WIDTH = 10               # 图片宽度
FIGURE_HEIGHT = 8              # 图片高度

# 🎯 散点图设置
POINT_SIZE1 = 100                 # 散点大小 (可调节: 10-200, 数值越大点越大)
POINT_SIZE2 = 300
POINT_ALPHA = 0.7               # 散点透明度
GRID_ALPHA = 0.4                # 网格透明度

# 💾 输出设置
OUTPUT_DPI = 1024                # 输出分辨率
OUTPUT_FORMAT = 'pdf'           # 输出格式: 'png', 'pdf', 'svg'

# 🔍 聚类设置
DEFAULT_N_CLUSTERS = 8          # 默认聚类数量
OUTLIER_PERCENTAGE = 0.85       # 边缘点百分比

# 🎯 两阶段删除设置
EDGE_DELETION_PERCENTAGE = 0.48   # 第一阶段：删除最边缘点的百分比（相对于总点数）
RANDOM_DELETION_PERCENTAGE = 0.45 # 第二阶段：随机删除点的百分比（相对于总点数）
# 示例：总共有100个点
# EDGE_DELETION_PERCENTAGE = 0.4 表示删除40个最边缘的点
# RANDOM_DELETION_PERCENTAGE = 0.3 表示再随机删除30个点
# 总删除率 = 0.4 + 0.3 = 0.7 (70%)

# 📊 数据处理设置
MAX_SAMPLES = 1000              # 最大处理样本数

# 🎨 更丰富的颜色选择 - 高对比度配色方案
colors_vibrant = [
    '#1f77b4',  # 深蓝色
    '#ff7f0e',  # 橙色  
    '#2ca02c',  # 绿色
    '#d62728',  # 红色 (如果不要可删除)
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 橄榄绿
    '#17becf'   # 青色
]

colors_no_red = [
    '#1f77b4',  # 深蓝色
    '#ff7f0e',  # 橙色  
    '#2ca02c',  # 绿色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#17becf',  # 青色
    '#bcbd22',  # 橄榄绿
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#aec7e8'   # 浅蓝色
]

colors_pastel = [
    '#AEC6CF',  # 浅蓝
    '#FFB347',  # 桃橙
    '#77DD77',  # 薄荷绿
    '#DDA0DD',  # 淡紫
    '#F49AC2',  # 粉红
    '#FFD1DC',  # 淡粉
    '#B39EB5',  # 淡紫灰
    '#CBC3E3',  # 薰衣草
    '#FDFD96',  # 浅黄
    '#87CEEB'   # 天空蓝
]

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

colors_rainbow = [
    '#FF0000',  # 红
    '#FF7F00',  # 橙
    '#FFFF00',  # 黄
    '#00FF00',  # 绿
    '#0000FF',  # 蓝
    '#4B0082',  # 靛
    '#9400D3',  # 紫
    '#FF1493',  # 深粉
    '#00CED1',  # 深青
    '#FFD700'   # 金
]

# 当前使用的颜色（你可以替换为上面任意一组）
colors_0 = colors_no_red

# 🔺 所有可用的形状选择 - 你可以从中挑选喜欢的
all_available_markers = {
    # === 基础几何形状 ===
    'o': '圆形 ●',
    's': '正方形 ■', 
    '^': '上三角 ▲',
    'v': '下三角 ▼',
    '<': '左三角 ◀',
    '>': '右三角 ▶',
    'D': '菱形 ♦',
    'd': '细菱形 ♢',
    
    # === 多边形 ===
    'p': '五角形 ⬟',
    'h': '六角形1 ⬡',
    'H': '六角形2 ⬢',
    '8': '八角形 ⯃',
    
    # === 特殊形状 ===
    '*': '星形 ★',
    'P': '加号(填充) ⊕',
    '+': '加号(线条) +',
    'x': 'X形 ×',
    'X': 'X形(粗) ✕',
    
    # === 线条形状 ===
    '|': '竖线 |',
    '_': '横线 _',
    
    # === 其他特殊形状 ===
    '1': '下花瓣 ⚘',
    '2': '上花瓣 ❀',
    '3': '左花瓣 ❁',
    '4': '右花瓣 ❂',
    '$...$': '数学符号 (需要LaTeX)',
}

# 🎯 推荐的形状组合（不同视觉效果）
markers_basic = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '+', 'x']
markers_geometric = ['o', 's', '^', 'v', 'D', 'd', 'p', 'h', 'H', '8']  
markers_mixed = ['o', 's', 'D', 'p', '*', 'h', '+', 'X', '8', 'v']
markers_special = ['*', 'p', 'h', 'H', '8', 'D', 'P', 'X', 'd', '+']

# 当前使用的形状（你可以替换为上面任意一组）
markers_0 = markers_basic

# ==================== 参数设置结束 ====================

# 设置matplotlib字体
import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.rcParams['font.family'] = FONT_FAMILY
    print(f"✅ 字体设置为: {FONT_FAMILY}")
except:
    print(f"⚠️ 字体 '{FONT_FAMILY}' 不可用")
    # matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def load_qa_data(file_path, max_samples=1000):
    """
    从JSON文件加载QA数据
    
    参数:
        file_path (str): JSON文件路径
        max_samples (int): 最大加载样本数，默认1000
    
    返回:
        list: 包含QA数据字典的列表，每个字典包含id、query、answer、type等字段
    """
    print(f"📁 正在从 {file_path} 加载QA数据...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，因为这是JSONL格式（每行一个JSON对象）
        lines = f.readlines()
    
    qa_data = []
    for i, line in enumerate(lines[:max_samples]):
        if line.strip():  # 跳过空行
            try:
                # 解析JSON数据
                data = json.loads(line.strip())
                qa_data.append({
                    'id': data.get('id', f'qa_{i}'),                                # QA对的唯一ID
                    'query': data['query'],                                         # 问题文本
                    'answer': data['answer_ground_truth'],                          # 标准答案
                    'type': data.get('metadata', {}).get('type', 'unknown'),       # 问题类型
                    'difficulty': data.get('metadata', {}).get('difficulty', 'hard') # 难度级别
                })
            except json.JSONDecodeError:
                print(f"⚠️  警告: 无法解析第 {i+1} 行")
                continue
    
    print(f"✅ 成功加载 {len(qa_data)} 个QA对")
    return qa_data

def generate_embeddings(qa_data, model_name='BAAI/bge-large-en-v1.5'):
    """
    为QA对的问题文本生成嵌入向量
    
    参数:
        qa_data (list): QA数据列表
        model_name (str): 嵌入模型名称
    
    返回:
        numpy.ndarray: 形状为(n_samples, embedding_dim)的嵌入矩阵
    """
    print(f"🤖 正在加载嵌入模型: {model_name}...")
    model = HuggingFaceEmbedding(model_name=model_name, device='cuda:2')
    
    print(f"🔄 正在为 {len(qa_data)} 个问题生成嵌入向量...")
    
    # 提取所有问题文本
    texts = [qa['query'] for qa in qa_data]
    
    # 批量生成嵌入向量（显示进度条）
    embeddings = model.get_text_embedding_batch(texts, show_progress=True)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"✅ 生成嵌入完成，形状: {embeddings_array.shape}")
    return embeddings_array

def reduce_to_2d_tsne(embeddings):
    """
    使用t-SNE将高维嵌入向量降维到2D
    
    参数:
        embeddings (numpy.ndarray): 高维嵌入向量矩阵
    
    返回:
        numpy.ndarray: 形状为(n_samples, 2)的2D坐标矩阵
    """
    print(f"📊 使用t-SNE将 {embeddings.shape[0]} 个嵌入向量降维到2D...")
    
    # 根据样本数量调整perplexity参数（t-SNE的重要超参数）
    # perplexity控制算法考虑每个点的近邻数量，通常设为5-50
    perplexity = min(30, max(5, len(embeddings) // 4))
    
    # 创建t-SNE对象
    tsne = TSNE(
        n_components=2,         # 降维目标维度：2D
        random_state=42,        # 随机种子，确保结果可复现
        perplexity=perplexity,  # 困惑度参数
        max_iter=1000,          # 最大迭代次数
        learning_rate='auto',   # 学习率自动设置
        init='pca'             # 初始化方法：使用PCA初始化
    )
    
    # 执行降维
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print(f"✅ t-SNE降维完成，2D坐标形状: {embeddings_2d.shape}")
    return embeddings_2d

def perform_clustering(embeddings_2d, n_clusters=8):
    """
    对2D嵌入向量执行K-means聚类分析
    
    参数:
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        n_clusters (int): 聚类数量，默认8个
    
    返回:
        numpy.ndarray: 每个样本的聚类标签数组
    """
    # 确保聚类数不超过样本数
    if len(embeddings_2d) < n_clusters:
        n_clusters = len(embeddings_2d)
    
    print(f"🔍 正在执行K-means聚类，聚类数: {n_clusters}...")
    
    # 创建K-means聚类器
    kmeans = KMeans(
        n_clusters=n_clusters,  # 聚类数量
        random_state=42,        # 随机种子，确保结果可复现
        n_init=10,             # 运行次数，选择最佳结果
        max_iter=250           # 最大迭代次数
    )
    
    # 执行聚类
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    print(f"✅ 聚类完成，发现 {len(np.unique(cluster_labels))} 个聚类")
    return cluster_labels

def analyze_clusters(qa_data, cluster_labels):
    """
    分析每个聚类的特征和内容
    
    参数:
        qa_data (list): QA数据列表
        cluster_labels (numpy.ndarray): 聚类标签数组
    
    返回:
        dict: 聚类分析结果字典
    """
    print(f"📈 正在分析聚类特征...")
    
    unique_clusters = np.unique(cluster_labels)
    cluster_analysis = {}
    
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的所有QA对
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_qa = [qa_data[i] for i in cluster_indices]
        
        # 统计问题类型分布
        type_counts = {}
        for qa in cluster_qa:
            qa_type = qa['type']
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
        
        # 找到主导类型
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else 'unknown'
        dominant_percentage = (type_counts.get(dominant_type, 0) / len(cluster_qa)) * 100
        
        # 存储分析结果
        cluster_analysis[cluster_id] = {
            'size': len(cluster_qa),
            'type_distribution': type_counts,
            'dominant_type': dominant_type,
            'dominant_percentage': dominant_percentage,
            'sample_queries': [qa['query'][:60] + '...' for qa in cluster_qa[:3]]  # 前3个样本
        }
    
    return cluster_analysis

def identify_outlier_points(embeddings_2d, cluster_labels, edge_deletion_percentage=0.4, random_deletion_percentage=0.3):
    """
    分两次消失识别每个聚类中的边缘点：先消失最边缘的点，再随机消失一些点
    
    参数:
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        cluster_labels (numpy.ndarray): 聚类标签数组
        edge_deletion_percentage (float): 第一阶段删除最边缘点的百分比，默认0.4（40%）
        random_deletion_percentage (float): 第二阶段随机删除点的百分比，默认0.3（30%）
    
    返回:
        numpy.ndarray: 布尔数组，True表示该点是异常值（应标记为灰色）
    """
    total_deletion_percentage = edge_deletion_percentage + random_deletion_percentage
    print(f"🎯 正在分两次消失识别每个聚类中的边缘点...")
    print(f"   第一阶段(最边缘): {edge_deletion_percentage*100:.0f}%")
    print(f"   第二阶段(随机): {random_deletion_percentage*100:.0f}%")
    print(f"   总删除率: {total_deletion_percentage*100:.0f}%")
    
    unique_clusters = np.unique(cluster_labels)
    outlier_mask = np.zeros(len(embeddings_2d), dtype=bool)
    
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的所有点
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) <= 1:
            continue  # 跳过只有一个点的聚类
        
        # 计算聚类中心（质心）
        cluster_center = np.mean(cluster_points, axis=0)
        
        # 计算每个点到聚类中心的距离
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        
        # 第一阶段：确定需要删除的最边缘点数
        n_edge_outliers = int(len(cluster_points) * edge_deletion_percentage)
        
        # 第二阶段：确定需要随机删除的点数
        n_random_outliers = int(len(cluster_points) * random_deletion_percentage)
        
        outlier_indices_in_cluster = []
        
        # 1. 找到距离最远的n_edge_outliers个点
        if n_edge_outliers > 0:
            edge_outlier_indices = np.argsort(distances)[-n_edge_outliers:]
            outlier_indices_in_cluster.extend(edge_outlier_indices)
        
        # 2. 从剩余的点中随机选择n_random_outliers个点
        if n_random_outliers > 0:
            # 找到尚未被选为边缘异常值的点的索引
            remaining_indices = [i for i in range(len(cluster_points)) 
                               if i not in outlier_indices_in_cluster]
            
            # 如果剩余点数不够，则全部选择
            n_random_outliers = min(n_random_outliers, len(remaining_indices))
            
            if n_random_outliers > 0:
                # 随机选择剩余的异常值点
                np.random.seed(42)  # 设置随机种子以确保结果可复现
                random_outlier_indices = np.random.choice(
                    remaining_indices, 
                    size=n_random_outliers, 
                    replace=False
                )
                outlier_indices_in_cluster.extend(random_outlier_indices)
        
        # 转换为在原始数组中的索引
        if outlier_indices_in_cluster:
            outlier_indices_global = cluster_indices[outlier_indices_in_cluster]
            # 标记这些点为异常值
            outlier_mask[outlier_indices_global] = True
    
    outlier_count = np.sum(outlier_mask)
    total_count = len(embeddings_2d)
    print(f"✅ 两次消失完成，共标记 {outlier_count} 个点为边缘点（占总数的 {outlier_count/total_count*100:.1f}%）")
    
    return outlier_mask

def create_cluster_visualization_with_outliers(qa_data, embeddings_2d, cluster_labels, outlier_mask, save_path):
    """
    创建按聚类着色的2D可视化图，直接删除边缘点而不显示，不同聚类使用不同的颜色和形状
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        cluster_labels (numpy.ndarray): 聚类标签数组
        outlier_mask (numpy.ndarray): 异常值掩码，True表示边缘点（将被删除）
        save_path (str): 图片保存路径
    
    返回:
        matplotlib.figure.Figure: 生成的图形对象
    """
    print(f"🎨 正在创建聚类可视化图（边缘点直接删除）...")
    
    # 创建图形，设置大小（使用配置参数）
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # 获取唯一聚类标签
    unique_clusters = np.unique(cluster_labels)
    
    # 只保留非异常点（删除边缘点）
    normal_mask = ~outlier_mask  # 正常点（非边缘点）
    kept_embeddings = embeddings_2d[normal_mask]
    kept_cluster_labels = cluster_labels[normal_mask]
    
    # 定义颜色和形状的映射 - 更多样化的形状和无红色的配色方案
    colors = colors_0
    markers = markers_0
    
    # 为每个聚类绘制不同颜色和形状的点（只绘制保留的点）
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = kept_cluster_labels == cluster_id
        if np.any(cluster_mask):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter_normal = plt.scatter(
                kept_embeddings[cluster_mask, 0],   # X坐标
                kept_embeddings[cluster_mask, 1],   # Y坐标
                c=color,                             # 颜色（固定色彩）
                marker=marker,                       # 形状
                s=POINT_SIZE2                         # 点的大小（使用配置参数）
            )
    
    # 设置坐标轴标签（使用配置的字体大小）
    # plt.xlabel('t-SNE Dimension 1', fontsize=FONT_SIZE_LABEL)
    # plt.ylabel('t-SNE Dimension 2', fontsize=FONT_SIZE_LABEL)
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # 隐藏坐标轴数字标注
    plt.xticks([])
    plt.yticks([])
    
    # 添加网格（使用配置参数）
    plt.grid(True, alpha=GRID_ALPHA)
    
    # 设置坐标轴范围，基于保留的点
    x_min, x_max = kept_embeddings[:, 0].min(), kept_embeddings[:, 0].max()
    y_min, y_max = kept_embeddings[:, 1].min(), kept_embeddings[:, 1].max()
    
    # 添加一些边距
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（使用配置参数）
    plt.savefig(save_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    print(f"💾 删除边缘点后的可视化图已保存到: {save_path}")
    
    return plt.gcf()

def create_cluster_visualization(qa_data, embeddings_2d, cluster_labels, save_path):
    """
    创建按聚类着色的2D可视化图，不同聚类使用不同的颜色和形状
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        cluster_labels (numpy.ndarray): 聚类标签数组
        save_path (str): 图片保存路径
    
    返回:
        matplotlib.figure.Figure: 生成的图形对象
    """
    print(f"🎨 正在创建聚类可视化图...")
    
    # 创建图形，设置大小（使用配置参数）
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # 获取唯一聚类标签
    unique_clusters = np.unique(cluster_labels)
    
    # 定义颜色和形状的映射 - 更多样化的形状和无红色的配色方案
    colors = colors_0
    markers = markers_0
    
    # 为每个聚类绘制不同颜色和形状的点
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        scatter = plt.scatter(
            embeddings_2d[cluster_mask, 0],     # X坐标
            embeddings_2d[cluster_mask, 1],     # Y坐标
            c=color,                             # 颜色（固定色彩）
            marker=marker,                       # 形状
            s=POINT_SIZE1                         # 点的大小（使用配置参数）
        )
    
    # 设置坐标轴标签（使用配置的字体大小）
    # plt.xlabel('t-SNE Dimension 1', fontsize=FONT_SIZE_LABEL)
    # plt.ylabel('t-SNE Dimension 2', fontsize=FONT_SIZE_LABEL)
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # 隐藏坐标轴数字标注
    plt.xticks([])
    plt.yticks([])
    
    # 添加网格（使用配置参数）
    plt.grid(True, alpha=GRID_ALPHA)
    
    # 设置坐标轴范围，确保与带异常点的图完全一致
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    # 添加一些边距
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（使用配置参数）
    plt.savefig(save_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    print(f"💾 聚类可视化图已保存到: {save_path}")
    
    return plt.gcf()

def print_detailed_cluster_analysis(qa_data, cluster_labels):
    """
    打印详细的聚类分析报告
    
    参数:
        qa_data (list): QA数据列表
        cluster_labels (numpy.ndarray): 聚类标签数组
    """
    print("\n" + "="*80)
    print("                          🔍 详细聚类分析报告")
    print("="*80)
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_qa = [qa_data[i] for i in cluster_indices]
        
        print(f"\n📍 聚类 {cluster_id} ({len(cluster_qa)} 个QA对)")
        print("-" * 50)
        
        # 统计问题类型分布
        type_counts = {}
        for qa in cluster_qa:
            qa_type = qa['type']
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
        
        # 显示类型分布
        print("📊 问题类型分布:")
        for qa_type, count in sorted(type_counts.items()):
            percentage = (count / len(cluster_qa)) * 100
            print(f"   {qa_type:20s}: {count:3d} 个 ({percentage:5.1f}%)")
        
        # 显示样本问题
        print("\n📝 样本问题:")
        sample_size = min(3, len(cluster_qa))
        for i in range(sample_size):
            qa = cluster_qa[i]
            print(f"   {i+1}. {qa['query'][:70]}...")
        
        if len(cluster_qa) > 3:
            print(f"   ... 还有 {len(cluster_qa) - 3} 个问题")
    
    print("="*80)

def main():
    """
    主函数：执行完整的聚类可视化流程
    """
    print("🚀 开始生成QA聚类可视化图...")
    print("=" * 60)
    
    # 1. 设置文件路径
    data_file = 'docs/dataset/unified/2wikimultihopqa_qa_unified.json'
    
    # 尝试多个可能的文件路径
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
            print(f"❌ 错误: 在以下路径都找不到数据文件:")
            print(f"   - {data_file}")
            for path in alternative_paths:
                print(f"   - {path}")
            return
    
    # 2. 设置参数（使用配置变量）
    max_samples = MAX_SAMPLES           # 处理的最大样本数
    n_clusters = DEFAULT_N_CLUSTERS     # 聚类数量
    edge_deletion_percentage = EDGE_DELETION_PERCENTAGE  # 第一阶段删除比率
    random_deletion_percentage = RANDOM_DELETION_PERCENTAGE  # 第二阶段删除比率
    # 3. 尝试使用本地模型（如果可用）
    try:
        local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if os.path.exists(local_model_path):
            model_name = local_model_path
            print(f"🔍 找到本地模型: {model_name}")
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
            print(f"🌐 使用在线模型: {model_name}")
    except:
        model_name = 'BAAI/bge-large-en-v1.5'
        print(f"🌐 使用在线模型: {model_name}")
    
    # 4. 加载QA数据
    qa_data = load_qa_data(data_file, max_samples=max_samples)
    
    if not qa_data:
        print("❌ 没有加载到QA数据，程序退出")
        return
    
    # 5. 生成嵌入向量
    embeddings = generate_embeddings(qa_data, model_name=model_name)
    
    # 6. 降维到2D
    embeddings_2d = reduce_to_2d_tsne(embeddings)
    
    # 7. 执行聚类分析
    cluster_labels = perform_clustering(embeddings_2d, n_clusters=n_clusters)
    
    # 8. 创建输出目录
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    
    # 9. 识别边缘点（使用配置的删除比率）
    outlier_mask = identify_outlier_points(
        embeddings_2d, 
        cluster_labels, 
        edge_deletion_percentage=edge_deletion_percentage,
        random_deletion_percentage=random_deletion_percentage
    )
    
    # 10. 创建并保存原始聚类可视化图
    save_path_original = f'{output_dir}/qa_embeddings_2d_clusters_{max_samples}.pdf'
    figure_original = create_cluster_visualization(qa_data, embeddings_2d, cluster_labels, save_path_original)
    
    # 11. 创建并保存带边缘点标记的可视化图
    save_path_outliers = f'{output_dir}/qa_embeddings_2d_clusters_outliers_{max_samples}.pdf'
    figure_outliers = create_cluster_visualization_with_outliers(qa_data, embeddings_2d, cluster_labels, outlier_mask, save_path_outliers)
    
    # 12. 打印详细的聚类分析报告
    print_detailed_cluster_analysis(qa_data, cluster_labels)
    
    print("=" * 60)
    print("🎉 聚类可视化完成！")
    print(f"📍 原始图片位置: {os.path.abspath(save_path_original)}")
    print(f"📍 删除边缘点后图片位置: {os.path.abspath(save_path_outliers)}")
    print(f"📊 处理了 {len(qa_data)} 个QA对")
    print(f"🔍 发现了 {len(np.unique(cluster_labels))} 个语义聚类")
    print(f"🎯 删除了 {np.sum(outlier_mask)} 个边缘点")
    print(f"   - 第一阶段删除最边缘的点: {edge_deletion_percentage*100:.0f}%")
    print(f"   - 第二阶段随机删除点: {random_deletion_percentage*100:.0f}%")
    print(f"   - 总删除率: {(edge_deletion_percentage + random_deletion_percentage)*100:.0f}%")
    print(f"🎨 生成了两个可视化图：原始聚类图 + 删除边缘点图")

if __name__ == '__main__':
    main()