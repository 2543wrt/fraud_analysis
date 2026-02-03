import networkx as nx
import numpy as np
from collections import Counter

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