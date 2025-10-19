import streamlit as st
import pandas as pd
import json
from utils.vector_search import SemanticMDM
from utils.taxonomy_viz import TaxonomyVisualizer
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Semantic MDM Demo - Chick-fil-A",
    page_icon="🐔",
    layout="wide"
)

# Initialize session state
if 'semantic_mdm' not in st.session_state:
    st.session_state.semantic_mdm = SemanticMDM()
    st.session_state.semantic_mdm.load_domain_data()

# Title
st.title("Semantic MDM: Aligning Legal & Restaurant Development")
st.markdown("### Breaking Down Data Silos Through AI-Powered Understanding")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose Demo Section:",
        ["The Problem", "The Solution", "Live Demo", "Architecture"]
    )
    
    st.markdown("---")
    st.markdown("**Demo for:** Chick-fil-A Interview")
    st.markdown("**Role:** Principal Legal & Restaurant Development Program Lead")

# Page 1: The Problem
if page == "The Problem":
    st.header("The Problem: Lost in Translation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Legal Department Says:")
        st.code("""
- NNN lease
- Ground lease agreement
- Tenant obligations
- Lease liability
- Property lease terms
- Real estate contract
        """, language="text")
    
    with col2:
        st.subheader("🏗️ Restaurant Development Says:")
        st.code("""
- Site lease
- Location agreement
- Development site terms
- Site agreement
- Property deal
- Real estate arrangement
        """, language="text")
    
    st.markdown("---")
    st.error("**Result**: Same concepts, different words → Systems can't connect them")
    
    # Show a scenario
    st.subheader("Real-World Scenario")
    st.info("""
    **Question from Legal**: "Show me all ground leases expiring in Q4 2025 in high-growth markets"
    
    **Current State**:
    - Legal searches their system for "ground leases" → finds 47 results
    - Restaurant Development searches for "site agreements" → finds 52 results  
    - Finance searches for "lease liabilities" → finds 49 results
    - Manual reconciliation takes 2-3 days
    - Risk of missing items or duplicates
    """)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Time Lost", "2-3 days", delta="-72 hours")
    col2.metric("Manual Touchpoints", "5-7 people", delta="+5")
    col3.metric("Error Risk", "High", delta="±15%")

# Page 2: The Solution
elif page == "The Solution":
    st.header("The Solution: Three-Layer Semantic MDM")
    
    # Layer 1: Taxonomy
    st.subheader("Layer 1️: Business Taxonomy (The Structure)")
    
    viz = TaxonomyVisualizer()
    fig = viz.create_taxonomy_tree()
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Purpose**: Creates canonical business concepts that all domains map to
    - Provides controlled vocabulary
    - Defines hierarchical relationships
    - Serves as the "source of truth" for business meaning
    """)
    
    st.markdown("---")
    
    # Layer 2: Conceptual DB
    st.subheader("Layer 2️: Conceptual Database (The Business Model)")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.code("""
LOCATION (Business Concept)
  Core Attributes:
    • location_id (master identifier)
    • address
    • coordinates
    • status
    
  Legal Perspective:
    • governed_by_agreement[]
    • legal_entity
    • jurisdiction
    • compliance_status
    
  Restaurant Development Perspective:
    • market_id
    • site_characteristics
    • development_stage
    • performance_projections
    
  Relationships:
    • has_agreements (1:many)
    • assigned_to_operator (1:1)
    • in_market (1:1)
        """, language="python")
    
    with col2:
        st.info("""
        **Key Insight**:
        
        Same business entity, different perspectives.
        
        Legal cares about compliance and contracts.
        
        RestDev cares about performance and market fit.
        
        **Both talk about the same LOCATION** - just different facets.
        """)
    
    st.markdown("---")
    
    # Layer 3: Vector DB
    st.subheader("Layer 3️: Vector Database (The Semantic Intelligence)")
    
    st.markdown("""
    **How it works**:
    1. Convert all terminology into vector embeddings (mathematical representations of meaning)
    2. When someone searches, find semantically similar terms across domains
    3. Auto-map to canonical taxonomy concepts
    4. Return unified results
    """)
    
    # Simple example
    st.code("""
User searches: "ground lease"

Vector DB finds similar terms:
  • NNN lease (Legal) → 0.94 similarity
  • site lease (RestDev) → 0.91 similarity  
  • location agreement (RestDev) → 0.87 similarity
  • property lease (Finance) → 0.85 similarity

All map to → Taxonomy: Agreements > Real Estate > Ground Lease
    """, language="text")

# Page 3: Live Demo
elif page == "Live Demo":
    st.header("Live Semantic Search Demo")
    
    # Search interface
    st.subheader("Try It: Cross-Domain Semantic Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter a search term (try 'ground lease', 'site agreement', 'operator'):",
            value="ground lease"
        )
    
    with col2:
        search_domain = st.selectbox(
            "Search from perspective of:",
            ["All Domains", "Legal", "Restaurant Development", "Finance"]
        )
    
    if st.button("🔍 Search", type="primary"):
        with st.spinner("Searching across domains..."):
            # Get semantic matches
            results = st.session_state.semantic_mdm.semantic_search(
                query=search_query,
                domain=search_domain,
                top_k=10
            )
            
            st.success(f"Found {len(results)} semantically related terms")
            
            # Display results
            st.subheader("Semantic Matches")
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Color code by domain
            def color_by_domain(domain):
                colors = {
                    'Legal': '🏛️',
                    'Restaurant Development': '🏗️',
                    'Finance': '💰',
                    'Operations': '⚙️'
                }
                return colors.get(domain, '📊')
            
            df['Domain_Icon'] = df['domain'].apply(color_by_domain)
            
            # Display table
            st.dataframe(
                df[['Domain_Icon', 'domain', 'term', 'similarity', 'canonical']].rename(columns={
                    'Domain_Icon': '',
                    'domain': 'Domain',
                    'term': 'Term Used',
                    'similarity': 'Similarity Score',
                    'canonical': 'Maps to Taxonomy'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            st.subheader("Semantic Similarity Map")
            
            fig = go.Figure()
            
            # Group by domain
            for domain in df['domain'].unique():
                domain_df = df[df['domain'] == domain]
                fig.add_trace(go.Bar(
                    name=domain,
                    x=domain_df['term'],
                    y=domain_df['similarity'],
                    text=[f"{s:.2%}" for s in domain_df['similarity']],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="How Similar Are These Terms to Your Search?",
                xaxis_title="Terms from Different Domains",
                yaxis_title="Semantic Similarity",
                yaxis_range=[0, 1],
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the magic
            st.markdown("---")
            st.subheader("🎯 The Insight")
            
            canonical = df['canonical'].iloc[0] if len(df) > 0 else "N/A"
            
            st.success(f"""
            **All these terms map to the same canonical concept**: `{canonical}`
            
            This means when Legal searches for "{search_query}", the system automatically:
            - Understands the semantic meaning
            - Finds related terms across ALL domains (not just exact matches)
            - Returns unified results
            - No manual mapping required!
            """)
    
    # Use Case Examples
    st.markdown("---")
    st.subheader("Try These Example Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Ground Lease"):
            st.info("Shows how Legal and RestDev use different terms for the same concept")
    
    with col2:
        if st.button("Development Site"):
            st.info("Shows how a RestDev term maps to Legal concepts")
    
    with col3:
        if st.button("Operator Agreement"):
            st.info("Shows cross-functional entity used by all domains")

# Page 4: Architecture
elif page == "Architecture":
    st.header("🏗️ Technical Architecture")
    
    st.subheader("System Components")
    
    # Architecture diagram (mermaid-style text)
    st.code("""
    ┌─────────────────────────────────────────────────┐
    │         User Interface (Streamlit)              │
    │  Legal Analyst │ RestDev Analyst │ Finance User │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │         Semantic Search Layer                    │
    │   • Query understanding                          │
    │   • Intent detection                             │
    │   • Cross-domain translation                     │
    └────────────┬────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼─────┐   ┌────▼──────┐
    │ Vector   │   │ Taxonomy  │
    │ Database │   │ Engine    │
    │          │   │           │
    │ Semantic │   │ Business  │
    │ Matching │   │ Rules     │
    └────┬─────┘   └────┬──────┘
         │              │
         └──────┬───────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │        Conceptual Database Layer                  │
    │  Entity Models │ Relationships │ Domain Views     │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │         Source Systems                            │
    │  Legal DB │ RestDev DB │ Finance DB │ Data Lake  │
    └──────────────────────────────────────────────────┘
    """, language="text")
    
    st.markdown("---")
    
    # Tech Stack
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🗄️ Data Layer")
        st.markdown("""
        **Taxonomy**
        - JSON-based hierarchy
        - Version controlled
        - Business-owned
        
        **Conceptual DB**
        - Graph database (Neo4j)
        - Flexible schema
        - Relationship-focused
        """)
    
    with col2:
        st.subheader("AI Layer")
        st.markdown("""
        **Vector Database**
        - Pinecone / Chroma
        - Real-time similarity
        - Scales with data
        
        **Embeddings**
        - Sentence-BERT
        - Domain-specific fine-tuning
        - OpenAI API (optional)
        """)
    
    with col3:
        st.subheader("🔌 Integration")
        st.markdown("""
        **Current Tools**
        - ThoughtSpot integration
        - Data Exchange API
        - Tableau connector
        
        **Access**
        - REST API
        - Python SDK
        - Self-service UI
        """)
    
    st.markdown("---")
    
    # Implementation Phases
    st.subheader("Implementation Roadmap")
    
    phases = pd.DataFrame({
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Timeline': ['Months 1-3', 'Months 4-6', 'Months 7-9', 'Months 10-12'],
        'Focus': [
            'Foundation: Build taxonomy, embed initial terms, prove concept',
            'Learning: Train vector DB, validate mappings, gather feedback',
            'Production: Integrate with tools, enable cross-domain queries',
            'Scale: Add domains, expand entities, advanced features'
        ],
        'Deliverables': [
            '• Top 5 entities modeled\n• Vector DB operational\n• Pilot with Legal + RestDev',
            '• 80%+ term coverage\n• Self-service search\n• User training complete',
            '• ThoughtSpot integration\n• API published\n• Usage analytics',
            '• 10+ entities\n• 5+ domains\n• Predictive capabilities'
        ]
    })
    
    for idx, row in phases.iterrows():
        with st.expander(f"**{row['Phase']}**: {row['Focus']}", expanded=(idx==0)):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Timeline", row['Timeline'])
            with col2:
                st.markdown(row['Deliverables'])
    
    st.markdown("---")
    
    # Success Metrics
    st.subheader("Success Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        label="Time to Insight",
        value="< 5 min",
        delta="-95%",
        delta_color="normal"
    )
    
    col2.metric(
        label="Cross-Domain Queries",
        value="500+/month",
        delta="+500",
        delta_color="normal"
    )
    
    col3.metric(
        label="Term Coverage",
        value="85%",
        delta="+85%",
        delta_color="normal"
    )
    
    col4.metric(
        label="User Satisfaction",
        value="4.5/5",
        delta="+2.5",
        delta_color="normal"
    )

# Footer
st.markdown("---")
st.markdown("Built with Love for Chick-fil-A • Demo by Andrew Pepper")