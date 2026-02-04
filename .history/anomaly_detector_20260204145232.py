import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

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
    
    def compute_risk_score(self, w_if=0.4, w_rf=0.6):
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
        
        risk_score = w_if * if_norm + w_rf * rf_norm
        
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

    def save_model(self, filepath):
        """Save the entire detector state to a file"""
        print(f"\nSaving model state to {filepath}...")
        joblib.dump(self, filepath)
        print("  ✓ Model saved successfully")