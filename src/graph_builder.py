import networkx as nx
import json
import os
from typing import Dict, List
from config import config

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_from_extraction(self, extraction_result: Dict) -> nx.DiGraph:
        """Build knowledge graph from entity extraction results with better error handling"""
        self.graph = nx.DiGraph()
        
        print("🕸️ Starting graph construction...")
        
        self._add_framework_node()
        
        sections_count = self._add_policy_sections(extraction_result)
        
        requirements_count = self._add_requirements(extraction_result)
        
        entities_count = self._add_entities(extraction_result)
        
        relationships_count = self._add_relationships(extraction_result)
        
        auto_relationships_count = self._add_auto_relationships(extraction_result)
        
        print(f"✅ Graph construction complete:")
        print(f"   - Sections: {sections_count}")
        print(f"   - Requirements: {requirements_count}")
        print(f"   - Entities: {entities_count}")
        print(f"   - Relationships: {relationships_count} (explicit) + {auto_relationships_count} (auto)")
        print(f"   - Total nodes: {self.graph.number_of_nodes()}")
        print(f"   - Total edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def _add_framework_node(self):
        """Add the main framework node"""
        self.graph.add_node(
            "Policy_Framework",
            type="framework",
            label="Document Knowledge Framework",
            description="Overall document knowledge structure"
        )
    
    def _add_policy_sections(self, extraction_result: Dict) -> int:
        """Add policy sections as nodes"""
        sections = extraction_result.get("policy_sections", [])
        sections_added = 0
        
        for section in sections:
            try:
                section_id = self._normalize_section_id(section["id"])
                
                self.graph.add_node(
                    section_id,
                    type="policy_section",
                    label=section.get("title", f"Section {section_id}"),
                    content=section.get("content", ""),
                    requirements_count=len(section.get("requirements", [])),
                    description=f"Policy section: {section.get('title', '')}"
                )
                
                self.graph.add_edge(
                    "Policy_Framework",
                    section_id,
                    relationship="contains_section",
                    description="Framework contains this section"
                )
                
                sections_added += 1
                
            except Exception as e:
                print(f"❌ Failed to add section {section.get('id', 'unknown')}: {e}")
        
        return sections_added
    
    def _add_requirements(self, extraction_result: Dict) -> int:
        """Add requirements as nodes"""
        requirements_added = 0
        sections = extraction_result.get("policy_sections", [])
        
        for section in sections:
            section_id = self._normalize_section_id(section["id"])
            requirements = section.get("requirements", [])
            
            for requirement in requirements:
                try:
                    req_id = self._normalize_requirement_id(requirement["id"])
                    
                    self.graph.add_node(
                        req_id,
                        type="requirement",
                        label=requirement.get("full_reference", req_id),
                        content=requirement.get("text", ""),
                        description=f"Requirement: {requirement.get('text', '')[:100]}..."
                    )
                    
                    self.graph.add_edge(
                        section_id,
                        req_id,
                        relationship="contains_requirement",
                        description=f"Section contains requirement {req_id}"
                    )
                    
                    requirements_added += 1
                    
                except Exception as e:
                    print(f"❌ Failed to add requirement {requirement.get('id', 'unknown')}: {e}")
        
        return requirements_added
    
    def _add_entities(self, extraction_result: Dict) -> int:
        """Add entities as nodes"""
        entities = extraction_result.get("entities", [])
        entities_added = 0
        
        for entity in entities:
            try:
                self.graph.add_node(
                    entity["id"],
                    type="entity",
                    label=entity["text"],
                    entity_type=entity.get("type", "ENTITY"),
                    description=entity.get("sentence", entity.get("description", "")),
                    original_text=entity["text"]
                )
                entities_added += 1
                
            except Exception as e:
                print(f"❌ Failed to add entity {entity.get('id', 'unknown')}: {e}")
        
        return entities_added
    
    def _add_relationships(self, extraction_result: Dict) -> int:
        """Add explicit relationships from extraction"""
        relationships = extraction_result.get("relationships", [])
        relationships_added = 0
        
        for relationship in relationships:
            try:
                from_node = relationship["from"]
                to_node = relationship["to"]
                
                if from_node in self.graph.nodes and to_node in self.graph.nodes:
                    self.graph.add_edge(
                        from_node,
                        to_node,
                        relationship=relationship.get("relationship", "related_to"),
                        description=relationship.get("text", "Relationship")
                    )
                    relationships_added += 1
                else:
                    print(f"⚠️ Skipping relationship {from_node} → {to_node}: nodes not found")
                    
            except Exception as e:
                print(f"❌ Failed to add relationship: {e}")
        
        return relationships_added
    
    def _add_auto_relationships(self, extraction_result: Dict) -> int:
        """Add automatic relationships based on content analysis"""
        auto_relationships_added = 0
        entities = extraction_result.get("entities", [])
        sections = extraction_result.get("policy_sections", [])
        
        for entity in entities:
            entity_text = entity["text"].lower()
            entity_id = entity["id"]
            
            for section in sections:
                section_id = self._normalize_section_id(section["id"])
                section_content = section.get("content", "").lower()
                
                if entity_text in section_content and section_id in self.graph.nodes:
                    self.graph.add_edge(
                        section_id,
                        entity_id,
                        relationship="mentions_entity",
                        description=f"Section mentions {entity['text']}"
                    )
                    auto_relationships_added += 1
            
            for section in sections:
                requirements = section.get("requirements", [])
                for requirement in requirements:
                    req_id = self._normalize_requirement_id(requirement["id"])
                    req_content = requirement.get("text", "").lower()
                    
                    if entity_text in req_content and req_id in self.graph.nodes:
                        self.graph.add_edge(
                            req_id,
                            entity_id,
                            relationship="involves_entity",
                            description=f"Requirement involves {entity['text']}"
                        )
                        auto_relationships_added += 1
        
        return auto_relationships_added
    
    def _normalize_section_id(self, section_id: str) -> str:

        if not section_id.startswith("Policy_Section_"):
            # Extract numbers from ID if possible
            numbers = ''.join(filter(str.isdigit, section_id))
            if numbers:
                return f"Policy_Section_{numbers}"
            else:
                return f"Policy_Section_{hash(section_id) % 10000}"
        return section_id
    
    def _normalize_requirement_id(self, requirement_id: str) -> str:
        if not requirement_id.startswith("Policy_Section_"):
            # Keep the original but ensure it's valid
            return requirement_id.replace(" ", "_").replace(".", "_")
        return requirement_id
    
    def save_graph(self, filename: str):
        """Save graph to JSON file"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                **node_data
            })
        
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })
        
        filepath = os.path.join(config.GRAPHS_DIR, f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, filename: str) -> nx.DiGraph:
        filepath = os.path.join(config.GRAPHS_DIR, f"{filename}.json")
        with open(filepath, "r") as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        
        for node in graph_data["nodes"]:
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        for edge in graph_data["edges"]:
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
        
        return self.graph
    
    def modify_node(self, node_id: str, new_details: str) -> bool:
        if node_id not in self.graph:
            return False
        
        if "content" in self.graph.nodes[node_id]:
            self.graph.nodes[node_id]["content"] = new_details
        else:
            self.graph.nodes[node_id]["description"] = new_details
        
        self.graph.nodes[node_id]["modified"] = True
        return True
    
    def delete_node(self, node_id: str) -> bool:
        if node_id not in self.graph:
            return False
        
        self.graph.remove_node(node_id)
        return True
    
    def get_node_info(self, node_id: str) -> Dict:
        if node_id not in self.graph:
            return {}
        
        node_data = self.graph.nodes[node_id].copy()
        node_data["id"] = node_id
        
        predecessors = list(self.graph.predecessors(node_id))
        successors = list(self.graph.successors(node_id))
        
        node_data["connected_to"] = {
            "incoming": predecessors,
            "outgoing": successors
        }
        
        return node_data