import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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