# Semantic MDM Demo - Cross-Domain Data Alignment

> **Breaking down data silos through AI-powered semantic understanding**

An interactive demonstration of how vector databases and semantic AI can align terminology across Legal, Restaurant Development, and Finance departments - enabling seamless cross-functional collaboration.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)

## The Problem

Different departments use different terminology for the same business concepts:

| Legal | Restaurant Development | Finance |
|-------|----------------------|---------|
| Ground lease | Site lease | Lease liability |
| NNN lease | Location agreement | Rent obligation |
| Operator contract | Franchisee assignment | Royalty agreement |

**Result**: Data silos, manual reconciliation, 2-3 day delays, risk of errors.

## The Solution

A three-layer semantic MDM architecture:

1. **Taxonomy Layer**: Canonical business concepts
2. **Conceptual Database**: Business entity models with multi-domain views
3. **Vector Database**: AI-powered semantic similarity matching

## ✨ Key Features

- **Semantic Search**: Find related terms across domains automatically
- **Auto-Mapping**: Terms automatically map to canonical concepts
- **Visual Analytics**: Interactive similarity scoring and visualization
- **Self-Learning**: System improves with usage
- **Business-Focused**: Solves real cross-functional challenges

## Quick Start

### Option 1: Using UV (Recommended - Faster)
```bash
 Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/YOUR_USERNAME/semantic-mdm-demo.git
cd semantic-mdm-demo
uv pip install -r requirements.txt
uv run streamlit run app.py
```

### Option 2: Using pip/venv
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/semantic-mdm-demo.git
cd semantic-mdm-demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure
```
semantic-mdm-demo/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── vector_search.py   # Semantic search engine
│   └── taxonomy_viz.py    # Visualization utilities
├── data/
│   ├── legal_terms.json   # Legal domain terminology
│   ├── restdev_terms.json # Restaurant Development terms
│   └── finance_terms.json # Finance domain terms
├── requirements.txt       # Python dependencies
└── README.md
```

## Demo Flow

1. **The Problem** - See the terminology disconnect
2. **The Solution** - Understand the 3-layer architecture
3. **Live Demo** - Try semantic search yourself
4. **Architecture** - Technical implementation details

### Try These Searches:

- `ground lease` - See Legal/RestDev alignment
- `operator` - See multi-domain entity
- `development site` - See RestDev to Legal mapping

## Technical Architecture
```
┌─────────────────────────────────────────────────┐
│         User Interface (Streamlit)              │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│         Semantic Search Layer                    │
│   • Query understanding                          │
│   • Cross-domain translation                     │
└────────────┬────────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼────┐   ┌───▼──────┐
│ Vector   │   │ Taxonomy  │
│ Database │   │ Engine    │
│          │   │           │
│ Semantic │   │ Business  │
│ Matching │   │ Rules     │
└──────────┘   └───────────┘
```

### Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Search**: Cosine similarity with scikit-learn
- **Visualization**: Plotly, NetworkX
- **Package Manager**: UV (optional, 10x faster than pip)

## Use Cases

### For Legal:
*"Show me all ground leases expiring in Q4 2025 in high-growth markets"*
- Currently: 2-3 days of manual reconciliation
- With MDM: Single query, instant results

### For Restaurant Development:
*"Which development sites have pending legal review?"*
- Currently: Email back-and-forth, multiple systems
- With MDM: Real-time integrated view

### Cross-Functional:
*"What's our development pipeline status including legal blockers?"*
- Currently: Manual coordination meetings
- With MDM: Automated dashboard

## Business Value

- **Time Savings**: 2-3 days → 5 minutes
- **Accuracy**: Eliminates manual mapping errors
- **Collaboration**: Breaks down data silos
- **Scalability**: Grows with new domains automatically

## Implementation Roadmap

| Phase | Timeline | Focus |
|-------|----------|-------|
| 1 | Months 1-3 | Foundation: Top 5 entities, proof of concept |
| 2 | Months 4-6 | Learning: Train vector DB, validate mappings |
| 3 | Months 7-9 | Production: Tool integration, self-service |
| 4 | Months 10-12 | Scale: Expand domains and entities |

## Contributing

This is a demonstration project. For production use, consider:
- Graph database (Neo4j) for conceptual layer
- Production vector database (Pinecone, Weaviate)
- Data governance framework
- Enterprise authentication

## License

MIT License - feel free to use and adapt

## Author

Created for Chick-fil-A Advanced Analytics & Automation team interview

**Contact**: Andrew Pepper - pepper_andrew@yahoo.com

## Acknowledgments

Built with modern data engineering best practices:
- Semantic search using state-of-the-art transformer models
- Vector embeddings for meaning representation
- Business-focused taxonomy design

---

*Demo showcasing strategic thinking about data alignment, modern technical capabilities (AI/ML, semantic search), and practical approach to cross-functional challenges.*