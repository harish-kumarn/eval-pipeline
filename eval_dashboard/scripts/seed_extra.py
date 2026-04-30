import sqlite3
import random
from datetime import datetime

def seed_more():
    conn = sqlite3.connect('metrics.db')
    cursor = conn.cursor()
    
    # Add 10 dummy experiments
    for i in range(10, 20):
        table_name = f"experiment_{i}"
        timestamp = datetime.now().isoformat()
        hybrid = random.choice([True, False])
        reform = random.choice([True, False])
        rerank = random.choice([True, False])
        mrr = random.uniform(0.4, 0.9)
        recall = random.uniform(0.5, 0.95)
        
        # Insert into index
        cursor.execute('''
            INSERT OR REPLACE INTO "index" 
            (table_name, timestamp, hybrid_search, query_reformulation, reranking, mrr_average, recall_average)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (table_name, timestamp, hybrid, reform, rerank, mrr, recall))
        
        # Create dummy table for the experiment
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cursor.execute(f'''
            CREATE TABLE "{table_name}" (
                id INTEGER PRIMARY KEY,
                question TEXT,
                mrr REAL,
                recall REAL
            )
        ''')
        cursor.execute(f'INSERT INTO "{table_name}" (question, mrr, recall) VALUES (?, ?, ?)', 
                       ("Dummy question", mrr, recall))
        
    conn.commit()
    conn.close()
    print("Seeded 10 more experiments.")

if __name__ == "__main__":
    seed_more()
