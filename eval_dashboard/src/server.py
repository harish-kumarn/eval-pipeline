import sqlite3
import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="RAG Metrics Dashboard Backend")

# Enable CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_PATH = os.getenv("DATABASE_PATH", "metrics.db")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/index")
async def get_index():
    try:
        with get_db_connection() as conn:
            rows = conn.execute('SELECT * FROM "index"').fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/experiment/{table_name}")
async def get_experiment(table_name: str):
    try:
        with get_db_connection() as conn:
            # Validate table_name exists in index to prevent injection
            exists = conn.execute('SELECT 1 FROM "index" WHERE table_name = ?', (table_name,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail=f"Experiment '{table_name}' not found")
            
            rows = conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
            return [dict(row) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/experiment/{table_name}/csv")
async def get_experiment_csv(table_name: str):
    try:
        with get_db_connection() as conn:
            # Validate table_name exists in index to prevent injection
            exists = conn.execute('SELECT 1 FROM "index" WHERE table_name = ?', (table_name,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail=f"Experiment '{table_name}' not found")
            
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            
            # Export to CSV in memory
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv"
            )
            response.headers["Content-Disposition"] = f"attachment; filename={table_name}.csv"
            return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/trends")
async def get_trends():
    """Returns all index rows for trend analysis."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute('SELECT * FROM "index" ORDER BY timestamp ASC').fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/topk")
async def get_topk_sensitivity():
    """Aggregates metrics by top_k across all experiments."""
    try:
        with get_db_connection() as conn:
            tables = conn.execute('SELECT table_name FROM "index"').fetchall()
            
            all_data = []
            for row in tables:
                table = row["table_name"]
                try:
                    # Check if top_k exists in this table
                    cursor = conn.execute(f'PRAGMA table_info("{table}")')
                    cols = [c[1] for c in cursor.fetchall()]
                    
                    if "top_k" not in cols:
                        continue
                        
                    # Only select columns that actually exist
                    available = [c for c in ["top_k", "mrr", "recall", "ndcg", "faithfulness", "answer_relevancy"] if c in cols]
                    if "top_k" not in available:
                        continue
                        
                    df = pd.read_sql_query(f'SELECT {", ".join(available)} FROM "{table}"', conn)
                    all_data.append(df)
                except Exception:
                    continue
            
            if not all_data:
                return []
                
            combined = pd.concat(all_data)
            if "top_k" not in combined.columns:
                return []
                
            stats = combined.groupby("top_k").mean().reset_index()
            stats = stats.replace({pd.NA: None, float('nan'): None})
            return stats.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/scatter")
async def get_scatter_data():
    """Provides samples/all question-level metrics for scatter plots."""
    try:
        with get_db_connection() as conn:
            tables = conn.execute('SELECT table_name FROM "index"').fetchall()
            
            all_questions = []
            target_cols = ["mrr", "recall", "faithfulness", "answer_correctness", "context_precision", "answer_relevancy"]
            
            for row in tables:
                table = row["table_name"]
                try:
                    cursor = conn.execute(f'PRAGMA table_info("{table}")')
                    cols = [c[1] for c in cursor.fetchall()]
                    
                    available = [c for c in target_cols if c in cols]
                    if not available:
                        continue
                        
                    query_cols = available + (["question", "type"] if all(c in cols for c in ["question", "type"]) else [])
                    
                    df = pd.read_sql_query(
                        f'SELECT {", ".join(set(query_cols))} FROM "{table}" LIMIT 100', 
                        conn
                    )
                    df["experiment"] = table
                    all_questions.append(df)
                except Exception:
                    continue
            
            if not all_questions:
                return []
                
            combined = pd.concat(all_questions)
            combined = combined.replace({pd.NA: None, float('nan'): None})
            return combined.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
