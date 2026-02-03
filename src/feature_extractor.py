import networkx as nx
import numpy as np
import pandas as pd

class GraphFeatureExtractor:
    """Extract features from transaction graph for ML"""
    
    def __init__(self, graph, df_transactions):
        self.G = graph.G
        self.df = df_transactions
        
    def extract_account_features(self):
        """Extract features for each account"""
        print("\nExtracting account features...")
        features = []
        
        account_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'account']
        
        for account in account_nodes:
            in_degree = self.G.in_degree(account)
            out_degree = self.G.out_degree(account)
            
            outgoing_txs = [data['amount'] for u, v, data in self.G.out_edges(account, data=True) 
                          if 'amount' in data]
            incoming_txs = [data['amount'] for u, v, data in self.G.in_edges(account, data=True) 
                          if 'amount' in data]
            
            total_out = sum(outgoing_txs) if outgoing_txs else 0
            total_in = sum(incoming_txs) if incoming_txs else 0
            
            features.append({
                'account': account,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': in_degree + out_degree,
                'degree_ratio': out_degree / max(in_degree, 1),
                'total_amount_out': total_out,
                'total_amount_in': total_in,
                'amount_balance': total_in - total_out,
                'avg_tx_out': np.mean(outgoing_txs) if outgoing_txs else 0,
                'avg_tx_in': np.mean(incoming_txs) if incoming_txs else 0,
                'std_tx_out': np.std(outgoing_txs) if len(outgoing_txs) > 1 else 0,
                'max_tx_out': max(outgoing_txs) if outgoing_txs else 0,
                'n_outgoing_txs': len(outgoing_txs),
                'n_incoming_txs': len(incoming_txs),
            })
        
        df_features = pd.DataFrame(features)
        
        print("  Computing centrality measures...")
        try:
            account_subgraph = self.G.subgraph(account_nodes)
            pagerank = nx.pagerank(account_subgraph, max_iter=50)
            df_features['pagerank'] = df_features['account'].map(pagerank).fillna(0)
            
            betweenness = nx.betweenness_centrality(account_subgraph, k=min(100, len(account_nodes)))
            df_features['betweenness'] = df_features['account'].map(betweenness).fillna(0)
        except:
            print("  ! Centrality computation failed")
            df_features['pagerank'] = 0
            df_features['betweenness'] = 0
        
        print("  Computing infrastructure features...")
        for idx, row in df_features.iterrows():
            account = row['account']
            
            devices = set()
            ips = set()
            
            for neighbor in self.G.successors(account):
                if self.G.nodes[neighbor].get('type') == 'device':
                    devices.add(neighbor)
                elif self.G.nodes[neighbor].get('type') == 'ip':
                    ips.add(neighbor)
            
            df_features.at[idx, 'n_unique_devices'] = len(devices)
            df_features.at[idx, 'n_unique_ips'] = len(ips)
            
            shared_device_accounts = 0
            for device in devices:
                shared_device_accounts += len(list(self.G.predecessors(device))) - 1
            
            df_features.at[idx, 'shared_device_score'] = shared_device_accounts
        
        print(f"  ✓ Extracted {len(df_features.columns)} features for {len(df_features)} accounts")
        
        return df_features
    
    def extract_temporal_features(self):
        """Extract time-based patterns"""
        print("\nExtracting temporal features...")
        
        temporal_features = []
        
        for account in self.df['from_account'].unique():
            account_txs = self.df[self.df['from_account'] == account].sort_values('timestamp')
            
            if len(account_txs) > 1:
                time_diffs = account_txs['timestamp'].diff().dt.total_seconds() / 3600
                time_diffs = time_diffs.dropna()
                
                temporal_features.append({
                    'account': account,
                    'tx_time_mean': time_diffs.mean() if len(time_diffs) > 0 else 0,
                    'tx_time_std': time_diffs.std() if len(time_diffs) > 0 else 0,
                    'tx_time_min': time_diffs.min() if len(time_diffs) > 0 else 0,
                    'rapid_tx_count': (time_diffs < 1).sum() if len(time_diffs) > 0 else 0,
                    'time_span_days': (account_txs['timestamp'].max() - 
                                     account_txs['timestamp'].min()).total_seconds() / 86400,
                })
        
        df_temporal = pd.DataFrame(temporal_features)
        print(f"  ✓ Extracted temporal features for {len(df_temporal)} accounts")
        
        return df_temporal
    
    def extract_all_features(self):
        """Extract complete feature set"""
        account_features = self.extract_account_features()
        temporal_features = self.extract_temporal_features()
        
        df_all = account_features.merge(temporal_features, on='account', how='left')
        df_all = df_all.fillna(0)
        
        print(f"\n✓ Total features extracted: {len(df_all.columns) - 1}")
        
        return df_all