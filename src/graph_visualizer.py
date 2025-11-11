import networkx as nx
from typing import Dict

class GraphVisualizer:
    def __init__(self):
        self.colors = {
            "framework": "blue",
            "policy_section": "green", 
            "requirement": "orange",
            "entity": "red"
        }
    
    def visualize_graph_text(self, graph: nx.DiGraph) -> str:
        """Generate text-based visualization of the graph"""
        output = []
        output.append("=" * 60)
        output.append("KNOWLEDGE GRAPH VISUALIZATION")
        output.append("=" * 60)
        
        nodes_by_type = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))
        
        for node_type, nodes in nodes_by_type.items():
            output.append(f"\n{node_type.upper()} NODES:")
            output.append("-" * 40)
            for node_id, node_data in nodes:
                label = node_data.get("label", node_id)
                output.append(f"  {node_id}: {label}")
                
                content = node_data.get("content", "")
                if content and len(content) > 50:
                    output.append(f"      Content: {content[:50]}...")
        
        output.append(f"\nRELATIONSHIPS:")
        output.append("-" * 40)
        for source, target, edge_data in graph.edges(data=True):
            rel_type = edge_data.get("relationship", "connected_to")
            output.append(f"  {source} --[{rel_type}]--> {target}")
        
        output.append("\n" + "=" * 60)
        return "\n".join(output)
    
    def get_graph_stats(self, graph: nx.DiGraph) -> Dict:
        """Get statistics about the graph"""
        nodes_by_type = {}
        for _, node_data in graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
        
        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "nodes_by_type": nodes_by_type,
            "is_connected": nx.is_weakly_connected(graph),
            "density": nx.density(graph)
        }