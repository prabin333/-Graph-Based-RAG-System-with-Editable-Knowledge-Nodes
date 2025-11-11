## Graph-Enhanced RAG System: Assignment 3

A sophisticated Retrieval-Augmented Generation (RAG) system that converts unstructured documents into a dynamic, queryable Knowledge Graph. This document outlines the system architecture and features.

## OBJECTIVE

The core objective is to create a Graph-Enhanced RAG system that performs the following:

1.  Converts uploaded documents into a structured Knowledge Graph.
2.  Allows users to query documents using natural language (Graph-based RAG).
3.  Visualizes entities and relationships for better understanding.
4.  Enables interactive modification (edit/delete) of specific nodes and relationships.

## KEY FEATURES

- **Document Processing:** Upload documents and extract key entities and relationships using an LLM.
- **Knowledge Graph:** Build and maintain an interactive graph structure from extracted data.
- **Intelligent Querying:** Answer natural language questions by traversing the underlying knowledge graph.
- **Dynamic Editing:** Modify or delete graph nodes and relationships in real-time.
- **Visualization:** Provide text-based graph visualization and statistical insights.

## QUICK START & INSTALLATION

**1. Installation Steps**

```
# Clone and setup environment

git clone <repository>
cd ASSIGNMENT_3
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Ensure model files are in ./model/ directory

# Run the main.py file

python main.py
```

## SYSTEM ARCHITECTURE

The project is structured modularly:

```
ASSIGNMENT_3/
├── main.py                    # CLI Interface
├── src/
│   ├── graph_rag.py          # Main System Orchestrator
│   ├── entity_extractor.py   # LLM Entity Extraction
│   ├── graph_builder.py      # Knowledge Graph Construction
│   ├── graph_visualizer.py   # Visualization & Statistics
│   └── config.py             # Configuration Management
├── data/                     # Document Storage
├── graphs/                   # Serialized Graph Storage
├── uploads/                  # User Upload Directory
├── model/                    # Local LLM Models
└── requirements.txt          # Python Dependencies
```

---

## PROCESSING PIPELINE

Document Upload -> Text Extraction -> LLM Analysis

Entity/Relationship Extraction -> Graph Construction

Query Processing -> Graph Traversal -> LLM Answer Generation

Graph Modifications -> Real-time Updates -> Persistence

## KEY DESIGN DECISIONS

### 1. Pure LLM-Based Extraction

- Uses Gemma LLM exclusively for entity/relationship extraction.
- Eliminates dependency on multiple NLP libraries.
- Provides consistent, high-quality extraction with better context understanding.

### 2. Hierarchical Graph Structure

The graph captures document structure and content relationships.

### 3. JSON-First Interface

- Standardized JSON input/output for all operations.
- Supports both single-line and multi-line JSON input.
- Robust JSON parsing with automatic error correction.

NOTE : IN THE OUTPUT FOLDER YOU CAN SEE THE OUTPUT IMAGE.
