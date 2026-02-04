
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import os
import argparse
import pandas as pd

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_production_data(filepath):
    """Load real transaction data from CSV/Parquet"""
    logger.info(f"Loading production data from {filepath}")
    # In a real implementation, this might connect to a SQL DB or Data Lake
    return pd.read_csv(filepath)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Money Laundering Detection System')
    parser.add_argument('--data', type=str, help='Path to input transaction CSV (Production Mode)', default=None)
    args = parser.parse_args()
    
    # 1. Generate data
    if args.data and os.path.exists(args.data):
        df_transactions = load_production_data(args.data)
    else:
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
    df_results = detector.compute_risk_score(w_if=0.3, w_rf=0.7) # Tuned weights
    
    # 6. Real-time scoring demo
    scorer = RealTimeScorer(graph, detector)
    
    print("\n" + "="*80)
    print("REAL-TIME TRANSACTION SCORING EXAMPLES")
    print("="*80)
    
    test_cases = [
        df_transactions.iloc[0], # Just take the first few if labels aren't present
        df_transactions.iloc[min(10, len(df_transactions)-1)],
    ]
    
    # If synthetic data, we can select specific labels
    if 'label' in df_transactions.columns:
        test_cases = [
            df_transactions[df_transactions['label'] == 'legitimate'].iloc[0],
            df_transactions[df_transactions['label'] == 'circular_flow'].iloc[0] if 'circular_flow' in df_transactions['label'].values else df_transactions.iloc[0]
        ]
    
    for i, tx in enumerate(test_cases, 1):
        result = scorer.score_transaction(
            tx['from_account'], tx['to_account'], tx['amount'],
            tx['device_id'], tx['ip_address']
        )
        
        label_display = tx.get('label', 'UNKNOWN').upper()
        print(f"\nTest Case {i}: {label_display}")
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
    data_dir = 'generated_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    csv_path = os.path.join(data_dir, 'risk_scores.csv')
    df_results.to_csv(csv_path, index=False)
    
    # Save the trained detector for the API
    model_path = os.path.join(data_dir, 'model.joblib')
    detector.save_model(model_path)
    
    print(f"\n✓ Detailed results saved to '{csv_path}'")
    print("✓ Visualizations saved to 'generated_images' folder")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()