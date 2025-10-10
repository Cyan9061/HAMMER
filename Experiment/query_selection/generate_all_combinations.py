"""
批量生成所有颜色和形状组合的聚类可视化图
生成10种颜色方案 × 10种形状方案 = 100种组合

作者: Claude Code Assistant
功能: 批量生成所有可能的颜色和形状组合的可视化图
输出目录: visualizations/sele/

新增特性：
- 5种新的暗色系颜色方案：midnight, gothic, storm, forest, navy
- 6种专为暗色系优化的标记方案：bold, sharp, solid, contrast, thick, angular
- 可调节的标记大小：POINT_SIZE参数 (10-500)
- 高分辨率输出：1024 DPI PDF格式
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
POINT_SIZE = 150                # 散点大小 (可调节: 10-500, 数值越大点越大)
POINT_ALPHA = 0.7               # 散点透明度
GRID_ALPHA = 0.4                # 网格透明度

# 💾 输出设置
OUTPUT_DPI = 1024                # 输出分辨率
OUTPUT_FORMAT = 'pdf'           # 输出格式: 'png', 'pdf', 'svg'

# 🔍 聚类设置
DEFAULT_N_CLUSTERS = 8          # 默认聚类数量
OUTLIER_PERCENTAGE = 0.85       # 边缘点百分比

# 🎯 两阶段删除设置
EDGE_DELETION_PERCENTAGE = 0.5   # 第一阶段：删除最边缘点的百分比（相对于总点数）
RANDOM_DELETION_PERCENTAGE = 0.4 # 第二阶段：随机删除点的百分比（相对于总点数）

# 📊 数据处理设置
MAX_SAMPLES = 1000              # 最大处理样本数

# 🎨 所有颜色方案
color_schemes = {
    'vibrant': [
        '#1f77b4',  # 深蓝色
        '#ff7f0e',  # 橙色  
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 橄榄绿
        '#17becf'   # 青色
    ],
    'no_red': [
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
    ],
    'pastel': [
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
    ],
    'deep': [
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
    ],
    'rainbow': [
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
    ],
    # ======================== 新增暗色系方案 ========================
    'midnight': [
        '#191970',  # 午夜蓝
        '#2F2F2F',  # 暗灰
        '#1C1C1C',  # 炭黑
        '#36454F',  # 炭灰蓝
        '#483C32',  # 塔upe
        '#301934',  # 深茄紫
        '#0F0F23',  # 墨蓝
        '#353839',  # 炮铜灰
        '#2C3539',  # 暗青灰
        '#1B1B1B'   # 夜黑
    ],
    'gothic': [
        '#000000',  # 纯黑
        '#800020',  # 勃艮第红
        '#2F1B14',  # 深褐
        '#654321',  # 深棕
        '#36213E',  # 深紫
        '#1A1A1A',  # 暗炭
        '#2E2E2E',  # 深灰
        '#722F37',  # 深酒红
        '#4A4A4A',  # 中灰
        '#1E1E1E'   # 煤黑
    ],
    'storm': [
        '#2F4F4F',  # 暗石板灰
        '#1C2833',  # 暗蓝灰
        '#5D6D7E',  # 青灰
        '#34495E',  # 蓝灰
        '#17202A',  # 深蓝黑
        '#273746',  # 暗蓝
        '#85929E',  # 浅蓝灰
        '#212F3D',  # 深暗蓝
        '#566573',  # 灰蓝
        '#1B2631'   # 夜空蓝
    ],
    'forest': [
        '#0B3D0B',  # 深森林绿
        '#2F4F2F',  # 暗橄榄绿
        '#1E3A1E',  # 深绿
        '#355E35',  # 暗翠绿
        '#006400',  # 暗绿
        '#1F2F1F',  # 深墨绿
        '#4F7942',  # 森林绿
        '#2D5016',  # 深橄榄
        '#3B5323',  # 暗松绿
        '#0D1F0D'   # 深夜绿
    ],
    'navy': [
        '#000080',  # 海军蓝
        '#191970',  # 午夜蓝
        '#1E3A8A',  # 深蓝
        '#1D4ED8',  # 蓝色
        '#0F172A',  # 暗蓝
        '#1E40AF',  # 深蓝色
        '#075985',  # 天蓝
        '#0C4A6E',  # 深天蓝
        '#082F49',  # 非常深蓝
        '#1E1B4B'   # 深靛蓝
    ]
}

# 🔺 所有形状方案
marker_schemes = {
    'basic': ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '+', 'x'],
    'geometric': ['o', 's', '^', 'v', 'D', 'd', 'p', 'h', 'H', '8'],
    'mixed': ['o', 's', 'D', 'p', '*', 'h', '+', 'X', '8', 'v'],
    'special': ['*', 'p', 'h', 'H', '8', 'D', 'P', 'X', 'd', '+'],
    # ======================== 新增暗色系专用标记方案 ========================
    'bold': ['s', 'D', 'H', '*', 'P', '^', 'v', 'p', '8', 'h'],         # 粗重标记，适合暗色背景
    'sharp': ['^', 'v', '<', '>', 'D', 'd', '1', '2', '3', '4'],        # 尖锐标记，增强对比
    'solid': ['s', 'D', 'p', 'h', 'H', '8', 'P', 'o', '^', 'v'],       # 实心标记，暗色可见
    'contrast': ['*', 'P', 'X', '+', 'D', 's', 'H', 'p', '8', '^'],    # 高对比度标记
    'thick': ['s', 'D', 'H', 'P', '*', 'p', '8', '^', 'v', 'h'],       # 粗线条标记
    'angular': ['^', 'v', '<', '>', 'D', 'd', 's', 'p', 'H', '8']      # 有棱角的标记
}

# 设置matplotlib字体
import matplotlib
try:
    matplotlib.rcParams['font.family'] = FONT_FAMILY
    print(f"✅ 字体设置为: {FONT_FAMILY}")
except:
    print(f"⚠️ 字体 '{FONT_FAMILY}' 不可用")

def load_qa_data(file_path, max_samples=1000):
    """从JSON文件加载QA数据"""
    print(f"📁 正在从 {file_path} 加载QA数据...")
    
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

def generate_embeddings(qa_data, model_name='BAAI/bge-large-en-v1.5'):
    """为QA对的问题文本生成嵌入向量"""
    print(f"🤖 正在加载嵌入模型: {model_name}...")
    model = HuggingFaceEmbedding(model_name=model_name, device='cuda:2')
    
    print(f"🔄 正在为 {len(qa_data)} 个问题生成嵌入向量...")
    
    texts = [qa['query'] for qa in qa_data]
    embeddings = model.get_text_embedding_batch(texts, show_progress=True)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"✅ 生成嵌入完成，形状: {embeddings_array.shape}")
    return embeddings_array

def reduce_to_2d_tsne(embeddings):
    """使用t-SNE将高维嵌入向量降维到2D"""
    print(f"📊 使用t-SNE将 {embeddings.shape[0]} 个嵌入向量降维到2D...")
    
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
    
    print(f"✅ t-SNE降维完成，2D坐标形状: {embeddings_2d.shape}")
    return embeddings_2d

def perform_clustering(embeddings_2d, n_clusters=8):
    """对2D嵌入向量执行K-means聚类分析"""
    if len(embeddings_2d) < n_clusters:
        n_clusters = len(embeddings_2d)
    
    print(f"🔍 正在执行K-means聚类，聚类数: {n_clusters}...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=250
    )
    
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    print(f"✅ 聚类完成，发现 {len(np.unique(cluster_labels))} 个聚类")
    return cluster_labels

def create_visualization_combination(embeddings_2d, cluster_labels, colors, markers, color_name, marker_name, save_path):
    """创建特定颜色和形状组合的可视化图"""
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    unique_clusters = np.unique(cluster_labels)
    
    # 为每个聚类绘制不同颜色和形状的点
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.scatter(
            embeddings_2d[cluster_mask, 0],
            embeddings_2d[cluster_mask, 1],
            c=color,
            marker=marker,
            s=POINT_SIZE,
            alpha=POINT_ALPHA
        )
    
    # 设置标题
    plt.title(f'{color_name.capitalize()} Colors + {marker_name.capitalize()} Markers', 
              fontsize=FONT_SIZE_TITLE, pad=20)
    
    # 设置坐标轴
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, alpha=GRID_ALPHA)
    
    # 设置坐标轴范围
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形以释放内存
    
    print(f"💾 已保存: {save_path}")

def main():
    """主函数：生成所有颜色和形状组合的可视化图"""
    print("🚀 开始批量生成所有组合的聚类可视化图...")
    print("=" * 80)
    
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
            print(f"❌ 错误: 找不到数据文件")
            return
    
    # 2. 设置模型
    try:
        local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if os.path.exists(local_model_path):
            model_name = local_model_path
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
    except:
        model_name = 'BAAI/bge-large-en-v1.5'
    
    # 3. 创建输出目录
    output_dir = 'visualizations/sele/dark'
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    
    # 4. 加载和处理数据（只需要执行一次）
    print("\n🔄 正在加载和处理数据...")
    qa_data = load_qa_data(data_file, max_samples=MAX_SAMPLES)
    
    if not qa_data:
        print("❌ 没有加载到QA数据，程序退出")
        return
    
    embeddings = generate_embeddings(qa_data, model_name=model_name)
    embeddings_2d = reduce_to_2d_tsne(embeddings)
    cluster_labels = perform_clustering(embeddings_2d, n_clusters=DEFAULT_N_CLUSTERS)
    
    # 5. 生成所有组合
    print(f"\n🎨 开始生成 {len(color_schemes)} × {len(marker_schemes)} = {len(color_schemes) * len(marker_schemes)} 种组合...")
    
    combination_count = 0
    total_combinations = len(color_schemes) * len(marker_schemes)
    
    for color_name, colors in color_schemes.items():
        for marker_name, markers in marker_schemes.items():
            combination_count += 1
            
            # 创建文件名
            filename = f"combination_{color_name}_{marker_name}_{MAX_SAMPLES}.{OUTPUT_FORMAT}"
            save_path = os.path.join(output_dir, filename)
            
            print(f"🎯 [{combination_count:2d}/{total_combinations}] 生成组合: {color_name} + {marker_name}")
            
            # 生成可视化
            create_visualization_combination(
                embeddings_2d, 
                cluster_labels, 
                colors, 
                markers, 
                color_name, 
                marker_name, 
                save_path
            )
    
    print("\n" + "=" * 80)
    print("🎉 所有组合生成完成！")
    print(f"📍 图片保存位置: {os.path.abspath(output_dir)}")
    print(f"📊 生成了 {total_combinations} 个可视化图")
    print(f"🎨 颜色方案: {list(color_schemes.keys())}")
    print(f"🔺 形状方案: {list(marker_schemes.keys())}")
    
    # 打印文件列表
    print("\n📋 生成的文件列表:")
    for color_name in color_schemes.keys():
        for marker_name in marker_schemes.keys():
            filename = f"combination_{color_name}_{marker_name}_{MAX_SAMPLES}.{OUTPUT_FORMAT}"
            print(f"   📄 {filename}")

if __name__ == '__main__':
    main()