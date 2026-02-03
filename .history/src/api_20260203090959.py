from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import networkx as nx
from .real_time_scorer import RealTimeScorer
from .transaction_graph import TransactionGraph

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Global variables to hold loaded models
scorer = None

class TransactionRequest(BaseModel):
    from_account: str
    to_account: str
    amount: float
    device_id: str
    ip_address: str

@app.on_event("startup")
def load_model():
    global scorer
    model_path = os.path.join("generated_data", "model.joblib")
    
    if not os.path.exists(model_path):
        print("WARNING: Model file not found. Please run main.py first.")
        return

    try:
        # Load the trained detector
        detector = joblib.load(model_path)
        
        # Reconstruct a minimal graph for scoring context
        # In a real production system, this would connect to a GraphDB (Neo4j)
        # Here we rebuild it from the training transactions stored in the detector
        graph = TransactionGraph()
        graph.build_from_transactions(detector.df_transactions)
        
        scorer = RealTimeScorer(graph, detector)
        print("✓ Model and Graph loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": scorer is not None}

@app.post("/score")
def score_transaction(tx: TransactionRequest):
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = scorer.score_transaction(
        tx.from_account,
        tx.to_account,
        tx.amount,
        tx.device_id,
        tx.ip_address
    )
    
    return result