import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import confusion_matrix

def create_visualizations(df_results, feature_importance, graph):
    """Create and save visualizations"""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Risk score distribution
    axes[0, 0].hist(df_results['risk_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.6, color='orange', linestyle='--', label='High Risk')
    axes[0, 0].axvline(0.8, color='red', linestyle='--', label='Critical Risk')
    axes[0, 0].set_xlabel('Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Risk Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Feature importance
    top_features = feature_importance.head(10)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Top 10 Feature Importance')
    axes[0, 1].grid(alpha=0.3)
    
    # Confusion matrix
    if 'true_label' in df_results.columns:
        cm = confusion_matrix(df_results['true_label'], df_results['rf_prediction'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
    
    # Risk vs volume
    axes[1, 1].scatter(df_results['total_amount_out'], df_results['risk_score'], 
                       alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Total Amount Out')
    axes[1, 1].set_ylabel('Risk Score')
    axes[1, 1].set_title('Risk Score vs Transaction Volume')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved risk_analysis.png")
    
    # High-risk network visualization
    print("  Creating network visualization...")
    high_risk_accounts = df_results.nlargest(20, 'risk_score')['account'].tolist()
    
    subgraph_nodes = set(high_risk_accounts)
    for account in high_risk_accounts:
        subgraph_nodes.update(graph.G.successors(account))
        subgraph_nodes.update(graph.G.predecessors(account))
    
    subgraph = graph.G.subgraph(subgraph_nodes)
    
    plt.figure(figsize=(16, 12))
    
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        node_type = graph.G.nodes[node].get('type', 'unknown')
        
        if node_type == 'account':
            if node in high_risk_accounts:
                node_colors.append('red')
                node_sizes.append(500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(200)
        elif node_type == 'device':
            node_colors.append('orange')
            node_sizes.append(150)
        elif node_type == 'ip':
            node_colors.append('green')
            node_sizes.append(150)
        else:
            node_colors.append('gray')
            node_sizes.append(100)
    
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, arrows=True, 
                           arrowsize=10, width=0.5)
    
    labels = {node: node for node in high_risk_accounts if node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    plt.title('High-Risk Account Network\n(Red: High Risk, Blue: Related, Orange: Devices, Green: IPs)', 
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('high_risk_network.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved high_risk_network.png")