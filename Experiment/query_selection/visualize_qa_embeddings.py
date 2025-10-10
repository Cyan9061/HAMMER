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

class QAEmbeddingVisualizer:
    """
    Visualizes QA pairs from 2wikimultihopqa dataset using embeddings in 2D space
    """
    
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        print(f"Loading embedding model: {model_name}...")
        self.model = HuggingFaceEmbedding(model_name=model_name)
        
    def load_qa_data(self, file_path, max_samples=1000):
        """
        Load QA data from JSON file
        """
        print(f"Loading QA data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines since it's a JSONL format (one JSON per line)
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
                        'difficulty': data.get('metadata', {}).get('difficulty', 'unknown')
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {i+1}")
                    continue
        
        print(f"Loaded {len(qa_data)} QA pairs")
        return qa_data
    
    def generate_embeddings(self, qa_data, embedding_mode='query'):
        """
        Generate embeddings for the QA pairs
        
        Args:
            qa_data: List of QA dictionaries
            embedding_mode: 'query', 'answer', or 'combined'
        """
        print(f"Generating embeddings for {len(qa_data)} QA pairs (mode: {embedding_mode})...")
        
        texts = []
        for qa in qa_data:
            if embedding_mode == 'query':
                texts.append(qa['query'])
            elif embedding_mode == 'answer':
                texts.append(qa['answer'])
            elif embedding_mode == 'combined':
                texts.append(f"Q: {qa['query']} A: {qa['answer']}")
            else:
                raise ValueError("embedding_mode must be 'query', 'answer', or 'combined'")
        
        # Generate embeddings in batches
        embeddings = self.model.get_text_embedding_batch(texts, show_progress=True)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def reduce_to_2d(self, embeddings, method='tsne'):
        """
        Reduce embeddings to 2D using t-SNE or other methods
        """
        print(f"Reducing {embeddings.shape[0]} embeddings to 2D using {method}...")
        
        if method == 'tsne':
            # Adjust perplexity based on the number of samples
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
        else:
            raise ValueError("Currently only 'tsne' method is supported")
        
        print(f"2D embeddings shape: {embeddings_2d.shape}")
        return embeddings_2d
    
    def perform_clustering(self, embeddings_2d, n_clusters=5):
        """
        Perform clustering on 2D embeddings
        """
        if len(embeddings_2d) < n_clusters:
            n_clusters = len(embeddings_2d)
        
        print(f"Performing clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_2d)
        
        return cluster_labels
    
    def create_visualization(self, qa_data, embeddings_2d, cluster_labels=None, 
                           save_path=None, show_labels=False, color_by='cluster'):
        """
        Create 2D visualization of QA embeddings
        
        Args:
            qa_data: Original QA data
            embeddings_2d: 2D coordinates
            cluster_labels: Cluster assignments
            save_path: Path to save the plot
            show_labels: Whether to show text labels for points
            color_by: 'cluster', 'type', or 'difficulty'
        """
        plt.figure(figsize=(16, 12))
        
        # Prepare colors and labels based on color_by parameter
        if color_by == 'cluster' and cluster_labels is not None:
            colors = cluster_labels
            unique_values = np.unique(cluster_labels)
            color_map = plt.cm.tab10
            title_suffix = f"Colored by Clusters ({len(unique_values)} clusters)"
        elif color_by == 'type':
            type_to_num = {}
            colors = []
            for qa in qa_data:
                qa_type = qa['type']
                if qa_type not in type_to_num:
                    type_to_num[qa_type] = len(type_to_num)
                colors.append(type_to_num[qa_type])
            unique_values = list(type_to_num.keys())
            color_map = plt.cm.Set1
            title_suffix = f"Colored by Question Type ({len(unique_values)} types)"
        elif color_by == 'difficulty':
            diff_to_num = {'easy': 0, 'medium': 1, 'hard': 2}
            colors = [diff_to_num.get(qa['difficulty'], 0) for qa in qa_data]
            unique_values = ['easy', 'medium', 'hard']
            color_map = plt.cm.viridis
            title_suffix = "Colored by Difficulty"
        else:
            colors = 'blue'
            color_map = None
            title_suffix = ""
        
        # Create scatter plot
        if color_map:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=colors, cmap=color_map, alpha=0.7, s=50)
            
            # Add colorbar/legend
            if color_by == 'cluster':
                plt.colorbar(scatter, label='Cluster')
            elif color_by in ['type', 'difficulty']:
                # Create custom legend
                import matplotlib.patches as mpatches
                legend_elements = [mpatches.Patch(color=color_map(i/len(unique_values)), 
                                                label=str(val)) 
                                 for i, val in enumerate(unique_values)]
                plt.legend(handles=legend_elements, title=color_by.capitalize(), 
                          bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       c=colors, alpha=0.7, s=50)
        
        # Add labels if requested (only for small datasets)
        if show_labels and len(qa_data) <= 50:
            for i, qa in enumerate(qa_data):
                plt.annotate(f"{i}: {qa['query'][:30]}...", 
                           (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.title(f'2D Visualization of {len(qa_data)} QA Pairs Embeddings\n{title_suffix}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"Total QA pairs: {len(qa_data)}\n"
        if cluster_labels is not None:
            stats_text += f"Clusters: {len(np.unique(cluster_labels))}\n"
        
        # Count by type
        type_counts = {}
        for qa in qa_data:
            qa_type = qa['type']
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
        
        stats_text += "Question Types:\n"
        for qa_type, count in sorted(type_counts.items()):
            stats_text += f"  {qa_type}: {count}\n"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
    
    def analyze_clusters(self, qa_data, cluster_labels):
        """
        Analyze the characteristics of each cluster
        """
        print("\n" + "="*60)
        print("                CLUSTER ANALYSIS")
        print("="*60)
        
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_qa = [qa_data[i] for i in cluster_indices]
            
            print(f"\nCluster {cluster_id} ({len(cluster_qa)} QA pairs):")
            print("-" * 40)
            
            # Analyze by type
            type_counts = {}
            for qa in cluster_qa:
                qa_type = qa['type']
                type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
            
            print("Question Types:")
            for qa_type, count in sorted(type_counts.items()):
                percentage = (count / len(cluster_qa)) * 100
                print(f"  {qa_type}: {count} ({percentage:.1f}%)")
            
            # Show sample questions
            print("\nSample Questions:")
            sample_size = min(3, len(cluster_qa))
            for i in range(sample_size):
                qa = cluster_qa[i]
                print(f"  {i+1}. {qa['query'][:80]}...")
        
        print("="*60)

def main():
    """Main function to run the QA embedding visualization"""
    
    # File path
    data_file = 'docs/dataset/unified/2wikimultihopqa_qa_unified.json'
    
    # Check if file exists
    if not os.path.exists(data_file):
        # Try alternative paths
        alternative_paths = [
            '../../docs/dataset/unified/2wikimultihopqa_qa_unified.json',
            '../../docs/dataset/unified/2wikimultihopqa_qa_unified.json'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                data_file = path
                break
        else:
            print(f"Error: Could not find the data file at any of the expected paths")
            return
    
    # Parameters
    max_samples = 1000
    embedding_mode = 'query'  # Options: 'query', 'answer', 'combined'
    n_clusters = 8
    
    # Try to use local model if available
    try:
        local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if os.path.exists(local_model_path):
            model_name = local_model_path
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
    except:
        model_name = 'BAAI/bge-large-en-v1.5'
    
    # Create visualizer
    visualizer = QAEmbeddingVisualizer(model_name=model_name)
    
    # Load data
    qa_data = visualizer.load_qa_data(data_file, max_samples=max_samples)
    
    if not qa_data:
        print("No QA data loaded. Exiting.")
        return
    
    # Generate embeddings
    embeddings = visualizer.generate_embeddings(qa_data, embedding_mode=embedding_mode)
    
    # Reduce to 2D
    embeddings_2d = visualizer.reduce_to_2d(embeddings, method='tsne')
    
    # Perform clustering
    cluster_labels = visualizer.perform_clustering(embeddings_2d, n_clusters=n_clusters)
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Create visualizations with different color schemes
    save_path_cluster = f'visualizations/qa_embeddings_2d_clusters_{max_samples}.png'
    save_path_type = f'visualizations/qa_embeddings_2d_types_{max_samples}.png'
    save_path_difficulty = f'visualizations/qa_embeddings_2d_difficulty_{max_samples}.png'
    
    # Visualization colored by clusters
    print("\nCreating cluster-based visualization...")
    visualizer.create_visualization(
        qa_data, embeddings_2d, cluster_labels, 
        save_path=save_path_cluster, color_by='cluster'
    )
    
    # Visualization colored by question type
    print("\nCreating type-based visualization...")
    visualizer.create_visualization(
        qa_data, embeddings_2d, cluster_labels, 
        save_path=save_path_type, color_by='type'
    )
    
    # Visualization colored by difficulty
    print("\nCreating difficulty-based visualization...")
    visualizer.create_visualization(
        qa_data, embeddings_2d, cluster_labels, 
        save_path=save_path_difficulty, color_by='difficulty'
    )
    
    # Analyze clusters
    visualizer.analyze_clusters(qa_data, cluster_labels)
    
    print(f"\nAll visualizations saved in 'visualizations/' directory")
    print(f"- Cluster view: {save_path_cluster}")
    print(f"- Type view: {save_path_type}")
    print(f"- Difficulty view: {save_path_difficulty}")

if __name__ == '__main__':
    main()