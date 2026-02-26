import networkx as nx
import json
import os

class GeneEditingKnowledgeGraph:
    def __init__(self, persistence_path="data/knowledge_base/kg.json"):
        self.graph = nx.DiGraph()
        self.persistence_path = persistence_path
        self._initialize_base_ontology()
        self.load_graph()

    def save_graph(self):
        """Persist the graph to disk."""
        data = nx.node_link_data(self.graph)
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Knowledge Graph saved to {self.persistence_path}")

    def load_graph(self):
        """Load the graph from disk if it exists."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                print(f"Loaded Knowledge Graph with {len(self.graph.nodes)} nodes from {self.persistence_path}")
            except Exception as e:
                print(f"Error loading Knowledge Graph: {e}. Starting with base ontology.")
        else:
            print("No existing Knowledge Graph found. Starting fresh.")

    def _initialize_base_ontology(self):
        """Initialize the base ontology for the Gene Editing Almanac."""
        # Base Technologies
        self.add_node("CRISPR KO", "Editing_Tool", category="DNA Double-strand break")
        self.add_node("Base Editing", "Editing_Tool", category="Precision Editing")
        self.add_node("Prime Editing", "Editing_Tool", category="Precision Editing")
        self.add_node("RNA editing", "Editing_Tool", category="RNA Editing")

        # Base Cas Types
        self.add_node("SpCas9", "Cas_Type")
        self.add_node("SaCas9", "Cas_Type")
        self.add_node("Cas12a", "Cas_Type")
        self.add_node("Cas13", "Cas_Type")

        # Base Mutation Types
        self.add_node("SNV", "Mutation_Type")
        self.add_node("insertion", "Mutation_Type")
        self.add_node("deletion", "Mutation_Type")

        # Base Delivery Systems
        self.add_node("AAV", "Delivery_System")
        self.add_node("LNP", "Delivery_System")
        self.add_node("electroporation", "Delivery_System")

        # Base Evidence Levels
        self.add_node("Level 1", "Evidence_Level", description="Clinical")
        self.add_node("Level 2", "Evidence_Level", description="Animal")
        self.add_node("Level 3", "Evidence_Level", description="In vitro")
        self.add_node("Level 4", "Evidence_Level", description="Predictive")

    def add_node(self, name, node_type, **kwargs):
        """Add a node to the graph with a specific type and attributes."""
        if not name or name == "Unknown":
            return
        self.graph.add_node(name, type=node_type, **kwargs)

    def add_edge(self, source, target, relation, **kwargs):
        """Add a directed edge between two nodes."""
        if not source or source == "Unknown" or not target or target == "Unknown":
            return
        self.graph.add_edge(source, target, relation=relation, **kwargs)

    def ingest_structured_data(self, data, study_id):
        """
        Ingest structured data extracted from literature into the Knowledge Graph.
        
        Expected data format:
        {
            "technology": "CRISPR KO",
            "cas_type": "SpCas9",
            "mutation_type": "SNV",
            "efficiency": "65%",
            "off_target_risk": "Low",
            "cell_type": "HEK293",
            "species": "Human",
            "delivery_system": "AAV",
            "evidence_level": "Level 3"
        }
        """
        # 1. Add Study Node
        self.add_node(study_id, "Study")

        # 2. Extract entities
        tech = data.get("technology")
        cas = data.get("cas_type")
        mut = data.get("mutation_type")
        cell = data.get("cell_type")
        species = data.get("species")
        delivery = data.get("delivery_system")
        evidence = data.get("evidence_level")
        
        efficiency = data.get("efficiency", "Unknown")
        off_target = data.get("off_target_risk", "Unknown")

        # 3. Add Entity Nodes (if they don't exist, they will be created)
        self.add_node(tech, "Editing_Tool")
        self.add_node(cas, "Cas_Type")
        self.add_node(mut, "Mutation_Type")
        self.add_node(cell, "Cell_Type")
        self.add_node(species, "Species")
        self.add_node(delivery, "Delivery_System")
        self.add_node(evidence, "Evidence_Level")

        # 4. Add Relationships
        # Technology uses Cas Type
        self.add_edge(tech, cas, "uses_cas")
        
        # Technology can fix Mutation Type
        self.add_edge(tech, mut, "can_fix")
        
        # Study uses Technology
        self.add_edge(study_id, tech, "evaluates_technology")
        
        # Study uses Delivery System
        self.add_edge(study_id, delivery, "uses_delivery")
        
        # Study conducted in Cell Type / Species
        self.add_edge(study_id, cell, "conducted_in_cell")
        self.add_edge(study_id, species, "conducted_in_species")
        
        # Study has Evidence Level
        self.add_edge(study_id, evidence, "has_evidence_level")

        # Complex Relationship: Technology efficiency in a specific study/context
        # We represent this as an edge from Technology to Study with attributes
        if tech and tech != "Unknown":
            self.add_edge(
                tech, 
                study_id, 
                relation="demonstrates_efficiency",
                efficiency=efficiency,
                off_target_risk=off_target,
                cell_type=cell,
                mutation_type=mut
            )

    def query_technology_capabilities(self, technology):
        """Query what mutations a technology can fix and its recorded efficiencies."""
        if technology not in self.graph:
            return None
            
        capabilities = {
            "can_fix": [],
            "studies": []
        }
        
        for neighbor in self.graph.successors(technology):
            edge_data = self.graph.get_edge_data(technology, neighbor)
            relation = edge_data.get("relation")
            
            if relation == "can_fix":
                capabilities["can_fix"].append(neighbor)
            elif relation == "demonstrates_efficiency":
                capabilities["studies"].append({
                    "study_id": neighbor,
                    "efficiency": edge_data.get("efficiency"),
                    "off_target_risk": edge_data.get("off_target_risk"),
                    "cell_type": edge_data.get("cell_type"),
                    "mutation_type": edge_data.get("mutation_type")
                })
                
        return capabilities

    def get_all_nodes_by_type(self, node_type):
        """Retrieve all nodes of a specific type."""
        return [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == node_type]

    def export_to_json(self, filepath):
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    kg = GeneEditingKnowledgeGraph()
    
    # Test ingestion
    sample_data = {
        "technology": "Base Editing",
        "cas_type": "SpCas9",
        "mutation_type": "SNV",
        "efficiency": "75%",
        "off_target_risk": "Low",
        "cell_type": "HEK293",
        "species": "Human",
        "delivery_system": "AAV",
        "evidence_level": "Level 3"
    }
    kg.ingest_structured_data(sample_data, "PMID:12345678")
    
    print(f"Knowledge Graph Initialized with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges.")
    print("\nCapabilities of Base Editing:")
    print(json.dumps(kg.query_technology_capabilities("Base Editing"), indent=2))
