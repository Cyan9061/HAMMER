"""
QA难度可视化脚本
专门用于生成2wikimultihopqa数据集中QA对的难度分布二维可视化图

作者: Claude Code Assistant
功能: 读取QA数据 -> 生成嵌入 -> t-SNE降维 -> 按难度着色可视化
输出: qa_embeddings_2d_difficulty_1000.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.manifold import TSNE
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
        list: 包含QA数据字典的列表，每个字典包含id、query、answer、difficulty等字段
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
                    'id': data.get('id', f'qa_{i}'),                                    # QA对的唯一ID
                    'query': data['query'],                                             # 问题文本
                    'answer': data['answer_ground_truth'],                              # 标准答案
                    'difficulty': data.get('metadata', {}).get('difficulty', 'hard')   # 难度级别：easy/medium/hard
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

def create_difficulty_visualization(qa_data, embeddings_2d, save_path):
    """
    创建按难度着色的2D可视化图
    
    参数:
        qa_data (list): QA数据列表
        embeddings_2d (numpy.ndarray): 2D坐标矩阵
        save_path (str): 图片保存路径
    
    返回:
        matplotlib.figure.Figure: 生成的图形对象
    """
    print(f"🎨 正在创建难度可视化图...")
    
    # 创建图形，设置大小为16x12英寸
    plt.figure(figsize=(16, 12))
    
    # 定义难度到数字的映射（用于着色）
    difficulty_to_num = {
        'easy': 0,      # 简单：绿色
        'medium': 1,    # 中等：黄色  
        'hard': 2       # 困难：红色
    }
    
    # 为每个QA对分配颜色编号
    colors = []
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    
    for qa in qa_data:
        difficulty = qa['difficulty']
        colors.append(difficulty_to_num.get(difficulty, 2))  # 默认为hard
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    # 创建散点图
    scatter = plt.scatter(
        embeddings_2d[:, 0],    # X坐标
        embeddings_2d[:, 1],    # Y坐标  
        c=colors,               # 颜色（按难度）
        cmap='viridis',         # 颜色映射：viridis（紫-蓝-绿-黄）
        alpha=0.7,              # 透明度
        s=50                    # 点的大小
    )
    
    # 创建自定义图例
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#440154', label=f'Hard ({difficulty_counts["hard"]})'),      # 深紫色
        mpatches.Patch(color='#31688e', label=f'Medium ({difficulty_counts["medium"]})'),  # 蓝色
        mpatches.Patch(color='#fde725', label=f'Easy ({difficulty_counts["easy"]})')       # 黄色
    ]
    
    # 添加图例到右上角
    plt.legend(handles=legend_elements, title='Difficulty Level', 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # 设置标题和轴标签
    plt.title(f'2D Visualization of {len(qa_data)} QA Pairs Embeddings\nColored by Difficulty Level', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 创建统计信息文本框
    stats_text = f"📊 数据统计\n"
    stats_text += f"总QA对数: {len(qa_data)}\n\n"
    stats_text += f"难度分布:\n"
    total = sum(difficulty_counts.values())
    for difficulty, count in difficulty_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        stats_text += f"  {difficulty.capitalize()}: {count} ({percentage:.1f}%)\n"
    
    # 添加统计信息文本框到左上角
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            fontsize=11, fontfamily='monospace')
    
    # 调整布局，确保图例不被裁剪
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 可视化图已保存到: {save_path}")
    
    return plt.gcf()

def main():
    """
    主函数：执行完整的可视化流程
    """
    print("🚀 开始生成QA难度可视化图...")
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
    
    # 7. 创建输出目录
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    
    # 8. 创建并保存可视化图
    save_path = f'{output_dir}/qa_embeddings_2d_difficulty_{max_samples}.pdf'
    figure = create_difficulty_visualization(qa_data, embeddings_2d, save_path)
    
    print("=" * 60)
    print("🎉 可视化完成！")
    print(f"📍 图片位置: {os.path.abspath(save_path)}")
    print(f"📊 处理了 {len(qa_data)} 个QA对")
    print(f"🎨 按难度级别着色显示在2D坐标系中")

if __name__ == '__main__':
    main()