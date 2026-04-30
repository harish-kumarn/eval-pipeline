# VisionRAG Evaluation Pipeline Architecture

## High-Level Flow Diagram

The following flowchart outlines the structural components and how data flows from the Evaluation Dashboard, to the Pipeline orchestrator, into the VisionRAG API, and finally through the metrics calculation.

```mermaid
flowchart TD
    subgraph Dashboard[Eval Dashboard / UI]
        direction TB
        Config[Configure Experiment Parameters]
        subgraph Toggles[Plug & Play Components]
            T1[Query Reformulation On/Off + Params]
            T2[Hybrid Search On/Off + Params]
            T3[Reranking On/Off + Params]
        end
        View[View Evaluation Results Dashboard]
        
        Config --> Toggles
    end

    subgraph EvalPipeline[Eval Pipeline Backend]
        direction TB
        ExpEngine[Experiment Runner]
        Dataset[(Evaluation Dataset \n Queries & Ground Truth)]
        MetricCalc[Metrics Engine]
        
        Dataset --> ExpEngine
    end

    subgraph VisionRAG[VisionRAG System]
        direction TB
        API[VisionRAG Endpoint URI]
        QR[Query Reformulation Module]
        HS[Hybrid Search Retriever]
        RR[Reranker Module]
        Gen[LLM Generation Engine]
        
        API --> QR
        QR --> HS
        HS --> RR
        RR --> Gen
    end

    %% System Interactions
    Toggles -- "Sends Config Payload JSON" --> ExpEngine
    ExpEngine -- "Sends Query + Config Flags" --> API
    
    Gen -- "Returns: \n1. Retrieved Context Chunks\n2. Generated Answer" --> MetricCalc
    ExpEngine -. "Sends Ground Truth Matches" .-> MetricCalc
    
    MetricCalc -- "Computes Retriever & \n Generation Metrics" --> View
```

---

## Execution Sequence

This sequence diagram details step-by-step how a single evaluation experiment runs:

```mermaid
sequenceDiagram
    autonumber
    participant D as Eval Dashboard
    participant EP as Eval Pipeline
    participant VR as VisionRAG URL
    participant M as Metrics Engine

    D->>EP: Init Experiment (Config: QR=ON, HS=ON, RR=OFF)
    loop For each record in Evaluation Dataset
        EP->>VR: HTTP POST (Raw Query + Component Toggles)
        
        Note over VR: Dynamic Pipeline Execution
        opt If Query Reformulation == ON
            VR->>VR: Reformulate Query
        end
        
        opt If Hybrid Search == ON
            VR->>VR: Dense + BM25 Search
        end
        opt If Hybrid Search == OFF
            VR->>VR: Dense Search Only
        end
        
        opt If Reranking == ON
            VR->>VR: Cross-Encoder/Cohere Rerank
        end
        
        VR->>VR: Generate Final Answer
        
        VR-->>EP: Response (Retrieved Context Chunks, Generated Answer)
        
        EP->>M: Forward Data + Ground Truth
        M->>M: Calc Retriever Metrics (MRR, NDCG, Recall@K)
        M->>M: Calc Generation Metrics (BLEU, Faithfulness, Answer Relevance)
    end
    M-->>D: Aggregate Results for Experiment
```

---

## Why Exactly Are We Doing This? (The Rationale)

Building the evaluation pipeline this way—with plug-and-play decoupling and separate metric scoring—solves several highly critical engineering challenges for RAG applications:

### 1. Component Ablation Studies (Isolating the value)
Modern RAG systems are complex. If your system's answers improve, you need to know **why**. Did the expensive Reranker fix the issue, or was it the Query Reformulation? By building these features as dynamic "toggles" (on/off buttons), you can run systematic *ablation studies* (A/B testing). You can scientifically prove the ROI (Return on Investment) of each specific component without having to manually change the VisionRAG source code between tests.

### 2. Identifying the "Failure Bottleneck" 
In RAG, bad answers happen for two main reasons:
*   **Retriever Failure:** The search engine didn't find the right document chunk. 
*   **Generation Failure (Hallucination):** The search engine found the right document, but the LLM ignored it or hallucinated a bad answer.

By specifically forcing VisionRAG to send back the *interim Retrieved Chunks* alongside the *Generated Answer*, the Metric Engine can score both independently. If the retrieval metrics are high but generation metrics are low, you know you need a better LLM prompt. If retrieval metrics are low, you know you need to tune your Hybrid Search.

### 3. Hyperparameter Optimization at Scale
Beyond simple "On/Off" states, passing parameters (like `top_k=10` or varying the weight of dense vs. sparse in hybrid search) straight from the dashboard allows non-engineers (or automated tuning scripts) to constantly calibrate the pipeline against a golden validation dataset to find the mathematically optimal configuration for VisionRAG.

### 4. Zero-Downtime Benchmarking
By decoupling the evaluation pipeline from the core application, you treat VisionRAG strictly as a black-box API. This ensures that benchmarking doesn't clutter or slow down production code, maintaining a strict boundary between testing protocols and your live environment.
