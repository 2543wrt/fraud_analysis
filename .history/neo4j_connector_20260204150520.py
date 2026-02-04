import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipe the database clean before loading new simulation data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j database cleared.")

    def load_transactions(self, df):
        """Bulk load transactions from pandas DataFrame"""
        logger.info(f"Loading {len(df)} transactions into Neo4j...")
        
        # Convert DataFrame to list of dictionaries for Cypher UNWIND
        # Fill NaNs to avoid Cypher errors
        records = df.fillna("").to_dict('records')
        
        query = """
        UNWIND $batch AS row
        
        // Create Accounts
        MERGE (source:Account {id: row.from_account})
        MERGE (target:Account {id: row.to_account})
        
        // Create Transaction Relationship
        CREATE (source)-[r:TRANSFERRED]->(target)
        SET r.amount = row.amount,
            r.label = row.label,
            r.timestamp = row.step  // Assuming 'step' or similar exists, otherwise remove
        
        // Link Source to Device (if exists)
        FOREACH (ignoreMe IN CASE WHEN row.device_id <> "" THEN [1] ELSE [] END | 
            MERGE (d:Device {id: row.device_id})
            MERGE (source)-[:USED_DEVICE]->(d)
        )
        
        // Link Source to IP (if exists)
        FOREACH (ignoreMe IN CASE WHEN row.ip_address <> "" THEN [1] ELSE [] END | 
            MERGE (ip:IPAddress {id: row.ip_address})
            MERGE (source)-[:USED_IP]->(ip)
        )
        """
        
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                session.run(query, batch=batch)
                
        logger.info("Neo4j loading complete.")