# Money Laundering Detection System

This project simulates and detects money laundering patterns in financial transactions using graph analysis and machine learning.

## Overview

The system generates synthetic transaction data containing both legitimate transactions and known money laundering patterns (like circular money flows, mule networks, and structuring). It then builds a network graph of these transactions to identify suspicious relationships and uses machine learning models to flag high-risk accounts.

## How It Works

1.  **Data Generation**: Creates fake accounts, devices, IP addresses, and transactions. It injects specific fraud patterns.
2.  **Graph Analysis**: Connects accounts based on money transfers and shared information (like using the same device). It looks for cycles (money going in a circle) and shared infrastructure.
3.  **Feature Extraction**: Calculates statistics for each account, such as total money sent/received, number of connections, and time between transactions.
4.  **Anomaly Detection**: Uses two machine learning models:
    - **Isolation Forest**: Finds statistical outliers (transactions that look very different from normal).
    - **Random Forest**: A supervised model trained to recognize specific fraud patterns.
5.  **Risk Scoring**: Combines the results to give every account and transaction a risk score.

## Project Structure

- `fraud_analysis.py`: The main script that runs the entire analysis.
- `data_generator.py`: Creates the synthetic dataset.
- `transaction_graph.py`: Builds the network graph and finds circular flows.
- `feature_extractor.py`: Turns graph data into numbers for the AI models.
- `anomaly_detector.py`: Contains the machine learning logic.
- `real_time_scorer.py`: Demonstrates how to score a single transaction instantly.
- `visualization.py`: Generates charts and network diagrams.
- `reporting.py`: Prints a summary report to the console.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries (pandas, numpy, networkx, scikit-learn, matplotlib, seaborn).
3.  Run the main script:
    ```bash
    python fraud_analysis.py
    ```

## Outputs

After running the script, the system creates two folders:

- **generated_data/**: Contains `risk_scores.csv` with detailed risk metrics for every account.
- **generated_images/**: Contains visualizations:
  - `risk_analysis.png`: Charts showing risk distribution and feature importance.
  - `high_risk_network.png`: A visual diagram of the high-risk accounts and their connections.
