import networkx as nx
from typing import List, Dict, Any

class GraphReasoner:
    """
    Advanced Graph Reasoning module for discovering implicit relationships and paths
    in the Gene Editing Knowledge Graph.
    """
    def __init__(self, kg_graph: nx.DiGraph):
        self.graph = kg_graph

    def find_meta_paths(self, start_node: str, end_node: str, max_depth: int = 3) -> List[List[str]]:
        """
        Finds paths between two entities (e.g., a Disease and a Gene, or an Author and a Technology).
        Used to explain "How is X related to Y?".
        """
        if start_node not in self.graph or end_node not in self.graph:
            return []
            
        try:
            # Find all simple paths (limited by length to avoid explosion)
            paths = list(nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_depth))
            return paths
        except Exception as e:
            print(f"Path finding failed: {e}")
            return []

    def recommend_related_technologies(self, target_entity: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Uses graph topology to recommend technologies related to a target (e.g., a specific Cell Type or Mutation).
        Logic: neighbors of neighbors within 'Editing_Tool' category.
        """
        if target_entity not in self.graph:
            return []
            
        recommendations = []
        # 1. Get direct neighbors (e.g., target_entity="T-cell" -> neighbors might be studies, or specific edits)
        neighbors = list(self.graph.neighbors(target_entity)) + list(self.graph.predecessors(target_entity))
        
        candidate_techs = {}
        
        for neighbor in neighbors:
            # 2. Who else is connected to these neighbors?
            second_neighbors = list(self.graph.neighbors(neighbor)) + list(self.graph.predecessors(neighbor))
            for sn in second_neighbors:
                # Check if this node is a Technology
                node_attrs = self.graph.nodes[sn]
                if node_attrs.get("type") == "Editing_Tool" and sn != target_entity:
                    if sn not in candidate_techs:
                        candidate_techs[sn] = {"count": 0, "path": neighbor}
                    candidate_techs[sn]["count"] += 1

        # Sort by connectivity count
        sorted_techs = sorted(candidate_techs.items(), key=lambda x: x[1]["count"], reverse=True)[:top_k]
        
        for tech, data in sorted_techs:
            recommendations.append({
                "technology": tech,
                "score": data["count"],
                "reason": f"Connected via {data['path']}"
            })
            
        return recommendations

    def detect_communities(self):
        """
        Detects clusters of related concepts (e.g., 'Base Editing' cluster vs 'Prime Editing' cluster).
        Useful for broad summarization.
        """
        # Simple connected components for now (or modularity-based if graph is complex)
        # Using undirected view for community detection
        undirected = self.graph.to_undirected()
        communities = list(nx.community.label_propagation_communities(undirected))
        return communities
