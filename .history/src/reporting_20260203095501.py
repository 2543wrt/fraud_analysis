from sklearn.metrics import precision_recall_fscore_support

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
              f"Amount=R{row['total_amount_out']:,.0f}, Txs={int(row['n_outgoing_txs'])}")
    
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