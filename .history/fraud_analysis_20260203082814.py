"""
Money Laundering Detection System - Standalone Script
Run with: python money_laundering_detector.py
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import warnings

from data_generator import TransactionGenerator
from transaction_graph import TransactionGraph
from feature_extractor import GraphFeatureExtractor
from anomaly_detector import AnomalyDetector
from real_time_scorer import RealTimeScorer
from visualization import create_visualizations
from reporting import generate_report

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("MONEY LAUNDERING DETECTION SYSTEM")
print("=" * 80)
print("\n✓ All libraries imported successfully\n")


class TransactionGenerator:
    """Generate synthetic transaction data with money laundering patterns"""
    
    def __init__(self, n_accounts=1000, n_devices=500, n_ips=300, n_merchants=200):
        self.n_accounts = n_accounts
        self.n_devices = n_devices
        self.n_ips = n_ips
        self.n_merchants = n_merchants
        
        # Generate entity IDs
        self.accounts = [f"ACC_{i:06d}" for i in range(n_accounts)]
        self.devices = [f"DEV_{i:05d}" for i in range(n_devices)]
        self.ips = [f"IP_{i//256}.{i%256}.{np.random.randint(0,256)}.{np.random.randint(0,256)}" 
                    for i in range(n_ips)]
        self.merchants = [f"MER_{i:05d}" for i in range(n_merchants)]
        
    def generate_legitimate_transactions(self, n_transactions=5000):
        """Generate normal transaction patterns"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(n_transactions):
            tx = {
                'transaction_id': f"TX_{i:08d}",
                'timestamp': base_time + timedelta(minutes=np.random.randint(0, 43200)),
                'from_account': np.random.choice(self.accounts),
                'to_account': np.random.choice(self.merchants),
                'amount': np.random.lognormal(mean=4, sigma=1.5),
                'device_id': np.random.choice(self.devices),
                'ip_address': np.random.choice(self.ips),
                'label': 'legitimate'
            }
            transactions.append(tx)
        
        return transactions
    
    def generate_circular_flow(self, n_patterns=20):
        """Generate circular money flow patterns (layering)"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)
        tx_id = 100000
        
        for pattern in range(n_patterns):
            circle_size = np.random.randint(3, 8)
            circle_accounts = np.random.choice(self.accounts, circle_size, replace=False)
            amount = np.random.uniform(5000, 50000)
            start_time = base_time + timedelta(hours=np.random.randint(0, 720))
            
            for i in range(circle_size):
                from_acc = circle_accounts[i]
                to_acc = circle_accounts[(i + 1) % circle_size]
                tx_amount = amount * np.random.uniform(0.85, 0.95)
                tx_time = start_time + timedelta(minutes=i*np.random.randint(10, 120))
                
                tx = {
                    'transaction_id': f"TX_{tx_id:08d}",
                    'timestamp': tx_time,
                    'from_account': from_acc,
                    'to_account': to_acc,
                    'amount': tx_amount,
                    'device_id': np.random.choice(self.devices),
                    'ip_address': np.random.choice(self.ips),
                    'label': 'circular_flow'
                }
                transactions.append(tx)
                tx_id += 1
                amount = tx_amount
        
        return transactions
    
    def generate_mule_network(self, n_networks=10):
        """Generate mule account network patterns"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)
        tx_id = 200000
        
        for network in range(n_networks):
            source = np.random.choice(self.accounts)
            n_mules = np.random.randint(5, 15)
            mules = np.random.choice(self.accounts, n_mules, replace=False)
            network_device = np.random.choice(self.devices)
            network_ip = np.random.choice(self.ips)
            start_time = base_time + timedelta(hours=np.random.randint(0, 720))
            
            # Distribution phase
            for i, mule in enumerate(mules):
                tx = {
                    'transaction_id': f"TX_{tx_id:08d}",
                    'timestamp': start_time + timedelta(minutes=i*5),
                    'from_account': source,
                    'to_account': mule,
                    'amount': np.random.uniform(8000, 9999),
                    'device_id': network_device,
                    'ip_address': network_ip,
                    'label': 'mule_network'
                }
                transactions.append(tx)
                tx_id += 1
            
            # Collection phase
            for i, mule in enumerate(mules):
                dest = np.random.choice(self.accounts)
                tx = {
                    'transaction_id': f"TX_{tx_id:08d}",
                    'timestamp': start_time + timedelta(hours=np.random.randint(12, 48)),
                    'from_account': mule,
                    'to_account': dest,
                    'amount': np.random.uniform(7500, 9500),
                    'device_id': network_device,
                    'ip_address': network_ip,
                    'label': 'mule_network'
                }
                transactions.append(tx)
                tx_id += 1
        
        return transactions
    
    def generate_structuring(self, n_patterns=15):
        """Generate structuring/smurfing patterns"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)
        tx_id = 300000
        
        for pattern in range(n_patterns):
            source = np.random.choice(self.accounts)
            n_splits = np.random.randint(8, 20)
            start_time = base_time + timedelta(hours=np.random.randint(0, 720))
            
            for i in range(n_splits):
                tx = {
                    'transaction_id': f"TX_{tx_id:08d}",
                    'timestamp': start_time + timedelta(hours=i),
                    'from_account': source,
                    'to_account': np.random.choice(self.accounts),
                    'amount': np.random.uniform(9000, 9999),
                    'device_id': np.random.choice(self.devices),
                    'ip_address': np.random.choice(self.ips),
                    'label': 'structuring'
                }
                transactions.append(tx)
                tx_id += 1
        
        return transactions
    
    def generate_all(self):
        """Generate complete dataset"""
        print("Generating transaction data...")
        transactions = []
        
        transactions.extend(self.generate_legitimate_transactions(5000))
        print(f"  ✓ Generated {len([t for t in transactions if t['label'] == 'legitimate'])} legitimate transactions")
        
        transactions.extend(self.generate_circular_flow(20))
        print(f"  ✓ Generated {len([t for t in transactions if t['label'] == 'circular_flow'])} circular flow transactions")
        
        transactions.extend(self.generate_mule_network(10))
        print(f"  ✓ Generated {len([t for t in transactions if t['label'] == 'mule_network'])} mule network transactions")
        
        transactions.extend(self.generate_structuring(15))
        print(f"  ✓ Generated {len([t for t in transactions if t['label'] == 'structuring'])} structuring transactions")
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nTotal transactions: {len(df)}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df


class TransactionGraph:
    """Multi-layer graph for transaction analysis"""
    
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.transaction_index = {}
        
    def build_from_transactions(self, df):
        """Construct graph from transaction DataFrame"""
        print("\nBuilding transaction graph...")
        
        for idx, row in df.iterrows():
            tx_id = row['transaction_id']
            
            self.G.add_node(row['from_account'], type='account')
            self.G.add_node(row['to_account'], type='account')
            self.G.add_node(row['device_id'], type='device')
            self.G.add_node(row['ip_address'], type='ip')
            
            self.G.add_edge(
                row['from_account'],
                row['to_account'],
                key=tx_id,
                amount=row['amount'],
                timestamp=row['timestamp'],
                label=row['label'],
                tx_id=tx_id
            )
            
            self.G.add_edge(row['from_account'], row['device_id'], 
                          key=f"{tx_id}_dev", relationship='uses_device')
            self.G.add_edge(row['from_account'], row['ip_address'], 
                          key=f"{tx_id}_ip", relationship='uses_ip')
            
            self.transaction_index[tx_id] = {
                'from': row['from_account'],
                'to': row['to_account'],
                'amount': row['amount'],
                'timestamp': row['timestamp']
            }
        
        print(f"  ✓ Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        node_types = Counter([self.G.nodes[n].get('type', 'unknown') for n in self.G.nodes()])
        print(f"  ✓ Node types: {dict(node_types)}")
        
    def detect_cycles(self, max_length=10):
        """Detect circular flows in the graph"""
        print("\nDetecting circular flows...")
        cycles = []
        
        account_graph = self.G.subgraph([n for n in self.G.nodes() 
                                         if self.G.nodes[n].get('type') == 'account'])
        
        try:
            simple_cycles = list(nx.simple_cycles(account_graph))
            cycles = [c for c in simple_cycles if len(c) <= max_length]
            
            print(f"  ✓ Found {len(cycles)} cycles")
            
            if cycles:
                cycle_lengths = [len(c) for c in cycles]
                print(f"  ✓ Cycle length distribution: min={min(cycle_lengths)}, "
                      f"max={max(cycle_lengths)}, mean={np.mean(cycle_lengths):.1f}")
        except:
            print("  ! Cycle detection failed (graph too large)")
        
        return cycles
    
    def find_shared_infrastructure(self):
        """Find accounts sharing devices or IPs (potential mule networks)"""
        print("\nDetecting shared infrastructure...")
        suspicious_groups = []
        
        device_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'device']
        
        for device in device_nodes:
            connected_accounts = [n for n in self.G.predecessors(device) 
                                if self.G.nodes[n].get('type') == 'account']
            
            if len(connected_accounts) > 5:
                suspicious_groups.append({
                    'type': 'device',
                    'infrastructure': device,
                    'accounts': connected_accounts,
                    'count': len(connected_accounts)
                })
        
        ip_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('type') == 'ip']
        
        for ip in ip_nodes:
            connected_accounts = [n for n in self.G.predecessors(ip) 
                                if self.G.nodes[n].get('type') == 'account']
            
            if len(connected_accounts) > 5:
                suspicious_groups.append({
                    'type': 'ip',
                    'infrastructure': ip,
                    'accounts': connected_accounts,
                    'count': len(connected_accounts)
                })
        
        print(f"  ✓ Found {len(suspicious_groups)} suspicious infrastructure groups")
        
        if suspicious_groups:
            top_groups = sorted(suspicious_groups, key=lambda x: x['count'], reverse=True)[:5]
            print("\n  Top suspicious groups:")
            for i, group in enumerate(top_groups, 1):
                print(f"    {i}. {group['type'].upper()}: {group['infrastructure']} "
                      f"({group['count']} accounts)")
        
        return suspicious_groups


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


class AnomalyDetector:
    """Multi-model anomaly detection system"""
    
    def __init__(self, df_features, df_transactions):
        self.df_features = df_features.copy()
        self.df_transactions = df_transactions.copy()
        self.models = {}
        self.scaler = StandardScaler()
        
        self.feature_cols = [col for col in df_features.columns if col != 'account']
        self.X = self.df_features[self.feature_cols].values
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def train_isolation_forest(self, contamination=0.1):
        """Train Isolation Forest for anomaly detection"""
        print("\nTraining Isolation Forest...")
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = model.fit_predict(self.X_scaled)
        anomaly_scores = model.score_samples(self.X_scaled)
        
        self.df_features['if_anomaly'] = predictions
        self.df_features['if_score'] = -anomaly_scores
        
        self.models['isolation_forest'] = model
        
        n_anomalies = (predictions == -1).sum()
        print(f"  ✓ Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.1f}%)")
        
        return predictions, anomaly_scores
    
    def create_ground_truth_labels(self):
        """Create labels based on known suspicious patterns"""
        account_labels = {}
        
        for account in self.df_features['account']:
            acct_txs = self.df_transactions[
                (self.df_transactions['from_account'] == account) |
                (self.df_transactions['to_account'] == account)
            ]
            
            labels = acct_txs['label'].unique()
            if any(l != 'legitimate' for l in labels):
                account_labels[account] = 1
            else:
                account_labels[account] = 0
        
        self.df_features['true_label'] = self.df_features['account'].map(account_labels)
        
        return self.df_features['true_label'].values
    
    def train_supervised_model(self):
        """Train supervised model using labeled data"""
        print("\nTraining supervised Random Forest...")
        
        y = self.create_ground_truth_labels()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred_proba = model.predict_proba(self.X_scaled)[:, 1]
        self.df_features['rf_score'] = y_pred_proba
        self.df_features['rf_prediction'] = (y_pred_proba > 0.5).astype(int)
        
        print(f"  ✓ Train accuracy: {train_score:.3f}")
        print(f"  ✓ Test accuracy: {test_score:.3f}")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 important features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        self.models['random_forest'] = model
        self.feature_importance = feature_importance
        
        return model, feature_importance
    
    def compute_risk_score(self):
        """Combine multiple signals into unified risk score"""
        print("\nComputing unified risk scores...")
        
        if 'if_score' in self.df_features.columns:
            if_scores = self.df_features['if_score']
            if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        else:
            if_norm = 0
        
        if 'rf_score' in self.df_features.columns:
            rf_norm = self.df_features['rf_score']
        else:
            rf_norm = 0
        
        risk_score = 0.4 * if_norm + 0.6 * rf_norm
        
        self.df_features['risk_score'] = risk_score
        
        def categorize_risk(score):
            if score > 0.8:
                return 'CRITICAL'
            elif score > 0.6:
                return 'HIGH'
            elif score > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        self.df_features['risk_category'] = self.df_features['risk_score'].apply(categorize_risk)
        
        print("  Risk distribution:")
        print(self.df_features['risk_category'].value_counts().sort_index())
        
        return self.df_features


class RealTimeScorer:
    """Real-time transaction risk scoring"""
    
    def __init__(self, graph, detector):
        self.graph = graph
        self.detector = detector
        self.alert_threshold = 0.7
        
    def score_transaction(self, from_account, to_account, amount, device_id, ip_address):
        """Score a new transaction in real-time"""
        risk_factors = []
        risk_score = 0.0
        
        if from_account in self.detector.df_features['account'].values:
            account_risk = self.detector.df_features[
                self.detector.df_features['account'] == from_account
            ]['risk_score'].values[0]
            
            risk_score += account_risk * 0.4
            
            if account_risk > 0.7:
                risk_factors.append(f"Source account has high risk score: {account_risk:.2f}")
        
        if 9000 <= amount <= 9999:
            risk_score += 0.3
            risk_factors.append("Amount just below $10k threshold (potential structuring)")
        
        if from_account in self.graph.G and to_account in self.graph.G:
            if self.graph.G.has_edge(to_account, from_account):
                risk_score += 0.4
                risk_factors.append("Circular transaction pattern detected")
        
        if device_id in self.graph.G:
            device_users = list(self.graph.G.predecessors(device_id))
            if len(device_users) > 5:
                risk_score += 0.2
                risk_factors.append(f"Device shared by {len(device_users)} accounts")
        
        if ip_address in self.graph.G:
            ip_users = list(self.graph.G.predecessors(ip_address))
            if len(ip_users) > 5:
                risk_score += 0.2
                risk_factors.append(f"IP shared by {len(ip_users)} accounts")
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= self.alert_threshold:
            action = "BLOCK"
        elif risk_score >= 0.5:
            action = "REVIEW"
        else:
            action = "APPROVE"
        
        return {
            'risk_score': risk_score,
            'action': action,
            'risk_factors': risk_factors,
            'details': {
                'from': from_account,
                'to': to_account,
                'amount': amount,
                'device': device_id,
                'ip': ip_address
            }
        }


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


def generate_report(df_transactions, df_results, cycles, suspicious_groups, scorer):
    """Generate summary report"""
    print("\n" + "="*80)
    print("MONEY LAUNDERING DETECTION SUMMARY REPORT")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Transactions: {len(df_transactions):,}")
    print(f"  Total Accounts: {len(df_results):,}")
    
    print(f"\nRisk Distribution:")
    for category in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = (df_results['risk_category'] == category).sum()
        pct = count / len(df_results) * 100
        print(f"  {category:10s}: {count:4d} accounts ({pct:5.1f}%)")
    
    print(f"\nTop 10 Highest Risk Accounts:")
    top_risks = df_results.nlargest(10, 'risk_score')[['account', 'risk_score', 'risk_category', 
                                                         'total_amount_out', 'n_outgoing_txs']]
    for idx, row in top_risks.iterrows():
        print(f"  {row['account']}: Score={row['risk_score']:.3f} ({row['risk_category']}), "
              f"Amount=${row['total_amount_out']:,.0f}, Txs={int(row['n_outgoing_txs'])}")
    
    print(f"\nPattern Detection Results:")
    print(f"  Circular Flows Detected: {len(cycles)}")
    print(f"  Suspicious Infrastructure Groups: {len(suspicious_groups)}")
    
    if 'true_label' in df_results.columns:
        precision, recall, f1, _ = precision_recall_fscore_support(
            df_results['true_label'], 
            df_results['rf_prediction'],
            average='binary'
        )
        print(f"\nModel Performance:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    print(f"\nRecommendations:")
    critical_accounts = df_results[df_results['risk_category'] == 'CRITICAL']
    print(f"  1. Immediate investigation required for {len(critical_accounts)} CRITICAL risk accounts")
    print(f"  2. Enhanced monitoring for {len(suspicious_groups)} infrastructure groups")
    print(f"  3. Review and update transaction limits for structuring prevention")
    print(f"  4. Deploy real-time scoring system with {scorer.alert_threshold} threshold")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    
    # 1. Generate data
    generator = TransactionGenerator()
    df_transactions = generator.generate_all()
    
    # 2. Build graph
    graph = TransactionGraph()
    graph.build_from_transactions(df_transactions)
    
    # 3. Detect patterns
    cycles = graph.detect_cycles()
    suspicious_groups = graph.find_shared_infrastructure()
    
    # 4. Extract features
    feature_extractor = GraphFeatureExtractor(graph, df_transactions)
    df_features = feature_extractor.extract_all_features()
    
    # 5. Train models
    detector = AnomalyDetector(df_features, df_transactions)
    detector.train_isolation_forest(contamination=0.15)
    model, feature_importance = detector.train_supervised_model()
    df_results = detector.compute_risk_score()
    
    # 6. Real-time scoring demo
    scorer = RealTimeScorer(graph, detector)
    
    print("\n" + "="*80)
    print("REAL-TIME TRANSACTION SCORING EXAMPLES")
    print("="*80)
    
    test_cases = [
        df_transactions[df_transactions['label'] == 'legitimate'].iloc[0],
        df_transactions[df_transactions['label'] == 'circular_flow'].iloc[0],
        df_transactions[df_transactions['label'] == 'mule_network'].iloc[0],
        df_transactions[df_transactions['label'] == 'structuring'].iloc[0],
    ]
    
    for i, tx in enumerate(test_cases, 1):
        result = scorer.score_transaction(
            tx['from_account'], tx['to_account'], tx['amount'],
            tx['device_id'], tx['ip_address']
        )
        
        print(f"\nTest Case {i}: {tx['label'].upper()}")
        print(f"Transaction: {tx['from_account']} → {tx['to_account']} (${tx['amount']:.2f})")
        print(f"Risk Score: {result['risk_score']:.3f}")
        print(f"Action: {result['action']}")
        if result['risk_factors']:
            print("Risk Factors:")
            for factor in result['risk_factors']:
                print(f"  • {factor}")
        print("-" * 80)
    
    # 7. Create visualizations
    create_visualizations(df_results, feature_importance, graph)
    
    # 8. Generate report
    generate_report(df_transactions, df_results, cycles, suspicious_groups, scorer)
    
    # 9. Save results
    df_results.to_csv('risk_scores.csv', index=False)
    print("\n✓ Detailed results saved to 'risk_scores.csv'")
    print("✓ Visualizations saved to 'risk_analysis.png' and 'high_risk_network.png'")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()