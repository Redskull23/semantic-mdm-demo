import plotly.graph_objects as go
import networkx as nx

class TaxonomyVisualizer:
    def __init__(self):
        self.taxonomy = self._build_sample_taxonomy()
    
    def _build_sample_taxonomy(self):
        """Build a sample business taxonomy"""
        return {
            "Root": {
                "Physical Assets": {
                    "Location": {
                        "Restaurant Location": ["Standalone", "Mall-based", "Drive-thru Only", "Mobile Pickup"],
                        "Development Site": ["Pre-construction", "Under Development", "Ready for Opening"]
                    },
                    "Market": ["Trade Area", "MSA", "Region"]
                },
                "Agreements": {
                    "Real Estate": ["Ground Lease", "Building Lease", "Lease Terms"],
                    "Franchise": ["Operator Agreement", "Territory Rights"],
                    "Development": ["Site Rights", "Construction Agreement"]
                },
                "Business Entities": {
                    "Operator": ["Individual Franchisee", "Multi-unit Operator"],
                    "Legal Entity": ["Operating Company", "Holding Company"],
                    "Vendor": ["Supplier", "Contractor"]
                }
            }
        }
    
    def create_taxonomy_tree(self):
        """Create an interactive tree visualization of the taxonomy"""
        
        # Build graph
        G = nx.DiGraph()
        
        def add_nodes(parent, children, level=0):
            if isinstance(children, dict):
                for key, value in children.items():
                    G.add_edge(parent, key)
                    add_nodes(key, value, level + 1)
            elif isinstance(children, list):
                for item in children:
                    G.add_edge(parent, item)
        
        add_nodes("Business Taxonomy", self.taxonomy)
        
        # Create hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Business Taxonomy Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig