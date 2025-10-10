import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from dotenv import load_dotenv
load_dotenv()
from types import SimpleNamespace
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Import the L2FaissSelector from the existing script
from run_selection_graph import L2FaissSelector

class DissimilarityGraphVisualizer:
    """
    Visualizes the dissimilarity graph constructed by L2FaissSelector
    """
    
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.selector = L2FaissSelector(model_name=model_name)
        
    def visualize_graph(self, dataset, num_queries_to_select, 
                       similarity_threshold=0.249, num_samples_per_node=150, 
                       exclude_top_k=50, save_path=None, show_edge_weights=True,
                       max_edges_to_show=50):
        """
        Visualize the dissimilarity graph with selected queries highlighted
        
        Args:
            dataset: List of query data objects
            num_queries_to_select: Number of queries to select (for highlighting)
            similarity_threshold: Threshold for considering queries dissimilar
            num_samples_per_node: Number of samples per node for graph construction
            exclude_top_k: Number of most similar queries to exclude
            save_path: Path to save the visualization
            show_edge_weights: Whether to show edge weights on the graph
            max_edges_to_show: Maximum number of edges to display (for clarity)
        """
        print(f"Starting visualization with {len(dataset)} queries...")
        
        # Get selected queries and the full graph
        selected_data, selected_indices, full_graph = self.selector.select_queries(
            dataset=dataset,
            num_queries_to_select=num_queries_to_select,
            similarity_threshold=similarity_threshold,
            num_samples_per_node=num_samples_per_node,
            exclude_top_k=exclude_top_k
        )
        
        if not selected_data or not full_graph:
            print("Error: No data selected or graph is empty")
            return
            
        print(f"Selected {len(selected_data)} queries out of {len(dataset)}")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add all nodes
        for i in range(len(dataset)):
            G.add_node(i, query=dataset[i].question[:50] + "..." if len(dataset[i].question) > 50 else dataset[i].question)
        
        # Add edges with weights, but limit the number for visualization clarity
        edge_list = []
        for i, neighbors in full_graph.items():
            for j, weight in neighbors.items():
                if i < j:  # Avoid duplicate edges
                    edge_list.append((i, j, weight))
        
        # Sort edges by weight (descending) and take the top ones
        edge_list.sort(key=lambda x: x[2], reverse=True)
        edges_to_show = edge_list[:max_edges_to_show]
        
        for i, j, weight in edges_to_show:
            G.add_edge(i, j, weight=weight)
        
        print(f"Displaying {len(edges_to_show)} edges out of {len(edge_list)} total edges")
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        
        # Use embedding-based layout for clustering similar nodes
        pos = self._create_embedding_layout(G, dataset, selected_indices)
        
        # Draw nodes
        node_colors = ['red' if i in selected_indices else 'lightblue' for i in G.nodes()]
        node_sizes = [800 if i in selected_indices else 300 for i in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges(data=True)
        if edges:
            edge_weights = [edge[2]['weight'] for edge in edges]
            
            # Normalize edge weights for better visualization
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            if max_weight > min_weight:
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 4 + 0.5 for w in edge_weights]
            else:
                normalized_weights = [1.0] * len(edge_weights)
            
            nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.6, edge_color='gray')
            
            # Show edge weights if requested
            if show_edge_weights and len(edges) <= 20:  # Only show weights if not too many edges
                edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # Draw node labels (query indices)
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # Create title and legend
        plt.title(f'Dissimilarity Graph Visualization\n'
                 f'Total Queries: {len(dataset)}, Selected: {len(selected_indices)} (10%)\n'
                 f'Similarity Threshold: {similarity_threshold}, Edges Shown: {len(edges_to_show)}',
                 fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Selected Queries'),
            Patch(facecolor='lightblue', alpha=0.7, label='Non-selected Queries'),
            Line2D([0], [0], color='gray', alpha=0.6, label='Dissimilarity Edges')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        # Print analysis
        self._print_analysis(dataset, selected_indices, full_graph, edges_to_show)
        
        return selected_data, selected_indices, full_graph
    
    def _create_embedding_layout(self, G, dataset, selected_indices):
        """
        Create a layout based on query embeddings using t-SNE for better clustering visualization
        """
        # Get embeddings from the selector
        embeddings = self.selector.embeddings
        
        if embeddings is None or len(embeddings) == 0:
            print("Warning: No embeddings available, using spring layout")
            return nx.spring_layout(G, k=3, iterations=50)
        
        # Use t-SNE to reduce embeddings to 2D for visualization
        print("Creating embedding-based layout using t-SNE...")
        
        # Filter embeddings to only include nodes in the graph
        node_indices = list(G.nodes())
        node_embeddings = embeddings[node_indices]
        
        # Apply t-SNE for dimensionality reduction
        if len(node_embeddings) > 1:
            # Adjust perplexity based on the number of samples
            perplexity = min(30, max(5, len(node_embeddings) // 4))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       max_iter=1000, learning_rate='auto', init='pca')
            pos_array = tsne.fit_transform(node_embeddings)
            
            # Create position dictionary
            pos = {}
            for i, node_idx in enumerate(node_indices):
                pos[node_idx] = pos_array[i]
        else:
            # Fallback for single node
            pos = {node_indices[0]: np.array([0, 0])}
        
        # Optionally, apply clustering to color-code regions
        if len(node_embeddings) > 3:
            try:
                # Perform clustering on the original embeddings
                n_clusters = min(5, len(node_embeddings) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(node_embeddings)
                
                # Store cluster information for potential future use
                self.cluster_labels = {node_indices[i]: cluster_labels[i] for i in range(len(node_indices))}
            except:
                pass
        
        return pos
    
    def _print_analysis(self, dataset, selected_indices, full_graph, edges_shown):
        """Print analysis of the graph and selection"""
        print("\n" + "="*60)
        print("                GRAPH ANALYSIS")
        print("="*60)
        
        total_nodes = len(dataset)
        selected_nodes = len(selected_indices)
        
        # Count total edges in full graph
        total_edges = 0
        for i, neighbors in full_graph.items():
            for j in neighbors:
                if i < j:
                    total_edges += 1
        
        print(f"Total Queries (Nodes): {total_nodes}")
        print(f"Selected Queries: {selected_nodes} ({selected_nodes/total_nodes*100:.1f}%)")
        print(f"Total Edges in Full Graph: {total_edges}")
        print(f"Edges Displayed: {len(edges_shown)}")
        
        # Analyze selected subgraph
        if selected_indices:
            subgraph_edges = 0
            subgraph_weights = []
            for i in selected_indices:
                for j, weight in full_graph.get(i, {}).items():
                    if j in selected_indices and i < j:
                        subgraph_edges += 1
                        subgraph_weights.append(weight)
            
            print(f"Edges in Selected Subgraph: {subgraph_edges}")
            if subgraph_weights:
                print(f"Average Weight in Selected Subgraph: {np.mean(subgraph_weights):.4f}")
                print(f"Weight Range in Selected Subgraph: {min(subgraph_weights):.4f} - {max(subgraph_weights):.4f}")
        
        # Show some example queries
        print("\nExample Selected Queries:")
        for i, idx in enumerate(selected_indices[:5]):
            query = dataset[idx].question
            print(f"  {idx}: {query[:80]}{'...' if len(query) > 80 else ''}")
        
        if len(selected_indices) > 5:
            print(f"  ... and {len(selected_indices) - 5} more")
        
        print("="*60)

def main():
    """Main function to run the visualization"""
    # Load data
    data_file_paths = [
        'docs/2wikimultihopqa_backup.json',
        '../../docs/2wikimultihopqa_backup.json',
        '../../docs/2wikimultihopqa.json'
    ]
    
    input_data = None
    for path in data_file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            print(f"Successfully loaded `{path}`.")
            # Take first 100 queries as requested
            input_data = all_data[:100]
            break
        except FileNotFoundError:
            continue
    
    if input_data is None:
        print("Warning: Real data files not found. Creating diverse dummy data...")
        input_data = [
            {'question': 'What is the capital of France and when was it established?'},
            {'question': 'How does photosynthesis work in plants?'},
            {'question': 'Who wrote Romeo and Juliet and when was it first performed?'},
            {'question': 'What are the main causes of climate change?'},
            {'question': 'How do neural networks learn from data?'},
            {'question': 'What is the largest planet in our solar system?'},
            {'question': 'Who invented the telephone and in what year?'},
            {'question': 'What are the health benefits of regular exercise?'},
            {'question': 'How is chocolate made from cocoa beans?'},
            {'question': 'What causes earthquakes and how are they measured?'},
            {'question': 'Who painted the Mona Lisa and where is it displayed?'},
            {'question': 'What is the process of protein synthesis in cells?'},
            {'question': 'How do computers process binary code?'},
            {'question': 'What are the main components of the immune system?'},
            {'question': 'Who discovered DNA structure and when?'},
            {'question': 'What causes ocean tides and how do they work?'},
            {'question': 'How is electricity generated in power plants?'},
            {'question': 'What are the symptoms of diabetes and how is it treated?'},
            {'question': 'Who composed the Four Seasons and what instruments are featured?'},
            {'question': 'What is the theory of relativity and who proposed it?'}
        ] * 5  # Repeat to get 100 diverse questions
    
    # Convert to objects
    data_objects = [SimpleNamespace(**d) for d in input_data]
    
    # Parameters for visualization
    num_queries_to_select = 10  # 10% of 100
    similarity_threshold = 0.7  # Higher threshold to capture more edges
    num_samples_per_node = 30   # Reduced for faster processing with 100 queries
    exclude_top_k = 10          # Reduced proportionally
    
    # Try to use local model if available, otherwise use default
    try:
        local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if os.path.exists(local_model_path):
            model_name = local_model_path
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
    except:
        model_name = 'BAAI/bge-large-en-v1.5'
    
    # Create visualizer
    visualizer = DissimilarityGraphVisualizer(model_name=model_name)
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    save_path = 'visualizations/dissimilarity_graph_100queries_10percent.png'
    
    # Run visualization
    print(f"Visualizing dissimilarity graph with {len(data_objects)} queries...")
    print(f"Selecting {num_queries_to_select} queries ({num_queries_to_select/len(data_objects)*100:.0f}%)")
    
    result = visualizer.visualize_graph(
        dataset=data_objects,
        num_queries_to_select=num_queries_to_select,
        similarity_threshold=similarity_threshold,
        num_samples_per_node=num_samples_per_node,
        exclude_top_k=exclude_top_k,
        save_path=save_path,
        show_edge_weights=True,
        max_edges_to_show=30  # Show top 30 edges for clarity
    )
    
    if result is not None:
        selected_data, selected_indices, full_graph = result
        print(f"\nVisualization complete! Graph saved to: {save_path}")
    else:
        print("\nVisualization completed but no data was returned.")

if __name__ == '__main__':
    main()