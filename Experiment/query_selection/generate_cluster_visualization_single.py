"""
QA聚类可视化脚本
专门用于生成2wikimultihopqa数据集中QA对的语义聚类二维可视化图

作者: Claude Code Assistant
功能: 读取QA数据 -> 生成嵌入 -> t-SNE降维 -> K-means聚类 -> 按聚类着色可视化
输出: qa_embeddings_2d_clusters_1000.png
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
    model = HuggingFaceEmbedding(model_name=model_name)
    
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
        max_iter=300           # 最大迭代次数
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

def create_cluster_visualization(qa_data, embeddings_2d, cluster_labels, save_path):
    """
    创建按聚类着色的2D可视化图
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        cluster_labels (numpy.ndarray): 聚类标签数组
        save_path (str): 图片保存路径
    
    返回:
        matplotlib.figure.Figure: 生成的图形对象
    """
    print(f"🎨 正在创建聚类可视化图...")
    
    # 创建图形，设置大小为16x12英寸
    plt.figure(figsize=(16, 12))
    
    # 获取唯一聚类标签
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # 使用tab10颜色映射（最多10种不同颜色）
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    # 创建散点图，每个聚类使用不同颜色
    scatter = plt.scatter(
        embeddings_2d[:, 0],        # X坐标
        embeddings_2d[:, 1],        # Y坐标
        c=cluster_labels,           # 颜色（按聚类）
        cmap='tab10',               # 颜色映射：tab10（10种区分度高的颜色）
        alpha=0.7,                  # 透明度
        s=50                        # 点的大小
    )
    
    # 设置标题和轴标签
    # plt.title(f'2D Visualization of {len(qa_data)} QA Pairs Embeddings\n'
    #          f'Colored by Clusters ({n_clusters} clusters found)', 
    #          fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # 2. 设置参数
    max_samples = 1000  # 处理的最大样本数
    n_clusters = 10      # 聚类数量
    
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
    
    # 9. 创建并保存聚类可视化图
    save_path = f'{output_dir}/qa_embeddings_2d_clusters_{max_samples}.pdf'
    figure = create_cluster_visualization(qa_data, embeddings_2d, cluster_labels, save_path)
    
    # 10. 打印详细的聚类分析报告
    print_detailed_cluster_analysis(qa_data, cluster_labels)
    
    print("=" * 60)
    print("🎉 聚类可视化完成！")
    print(f"📍 图片位置: {os.path.abspath(save_path)}")
    print(f"📊 处理了 {len(qa_data)} 个QA对")
    print(f"🔍 发现了 {len(np.unique(cluster_labels))} 个语义聚类")
    print(f"🎨 按聚类ID着色显示在2D坐标系中")

if __name__ == '__main__':
    main()