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