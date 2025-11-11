#!/usr/bin/env python3
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph_rag import GraphRAGSystem

class GraphRAGCLI:
    def __init__(self):
        self.rag_system = GraphRAGSystem()
    
    def run(self):
        print("🧠 Graph-Based RAG System - Assignment 3")
        print("=" * 60)
        print("Supports JSON input format as per assignment requirements")
        print("=" * 60)
        print("📝 USAGE: First select option (1-7), THEN enter JSON when prompted")
        print("✅ Accepts both single-line and multi-line JSON")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. 📄 Upload and process document (JSON input)")
            print("2. ❓ Query knowledge graph (JSON input)") 
            print("3. ✏️  Modify graph node (JSON input)")
            print("4. 🗑️  Delete graph node (JSON input)")
            print("5. 📊 View graph visualization")
            print("6. 📈 Get graph statistics")
            print("7. 🚪 Exit")
            
            choice = input("\n👉 Select option (1-7): ").strip()
            
            if choice == "1":
                self.process_document_json()
            elif choice == "2":
                self.query_graph_json()
            elif choice == "3":
                self.modify_node_json()
            elif choice == "4":
                self.delete_node_json()
            elif choice == "5":
                self.visualize_graph()
            elif choice == "6":
                self.show_statistics()
            elif choice == "7":
                print("👋 Exiting Graph RAG System...")
                break
            else:
                print("❌ Invalid option. Please enter only 1-7.")
    
    def _get_json_input(self):
        print("👉 Enter JSON (press Enter twice when done for multi-line):")
        lines = []
        while True:
            try:
                line = input()
                if not line: 
                    break
                lines.append(line)
            except EOFError:
                break
        
        json_input = ' '.join(lines).strip()
        
        if not json_input:
            json_input = input("👉 Or enter single-line JSON: ").strip()
        
        return json_input
    
    def _parse_json_input(self, json_input):
        if not json_input:
            raise ValueError("JSON input is required")
        
        json_input = ' '.join(json_input.split())
        
        try:
            return json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def process_document_json(self):
        print("\n📄 PROCESS DOCUMENT")
        print("Supported JSON formats:")
        print('Single-line: {"document": "data/sample_policy.txt", "query": "List requirements"}')
        print('Multi-line:')
        print('  {')
        print('    "document": "policy_document.pdf",')
        print('    "query": "List the main compliance requirements."')
        print('  }')
        
        try:
            json_input = self._get_json_input()
            data = self._parse_json_input(json_input)
            
            document_path = data.get("document", "").strip()
            query = data.get("query", "").strip()
            
            if not document_path:
                print("❌ 'document' field is required in JSON.")
                return
            
            print(f"\n🔄 Processing document: {document_path}")
            result = self.rag_system.process_document_with_query(data)
            
            print("\n✅ RESULT:")
            print(json.dumps(result, indent=2))
            
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Error processing document: {e}")
    
    def query_graph_json(self):
        print("\n❓ QUERY KNOWLEDGE GRAPH")
        print('Single-line: {"query": "List the main compliance requirements."}')
        print('Multi-line:')
        print('  {')
        print('    "query": "List the main compliance requirements."')
        print('  }')
        
        try:
            json_input = self._get_json_input()
            data = self._parse_json_input(json_input)
            
            query = data.get("query", "").strip()
            
            if not query:
                print("❌ 'query' field is required in JSON.")
                return
            
            print(f"\n🔄 Processing query: {query}")
            result = self.rag_system.query_graph(query)
            
            print("\n✅ RESULT:")
            print(json.dumps(result, indent=2))
            
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Error querying graph: {e}")
    
    def modify_node_json(self):
        print("\n✏️ MODIFY GRAPH NODE")
        print('Single-line: {"action": "modify", "node_id": "Policy_Section_5.1", "new_details": "Updated requirements..."}')
        print('Multi-line:')
        print('  {')
        print('    "action": "modify",')
        print('    "node_id": "Policy_Section_5.1",')
        print('    "new_details": "Updated requirements include risk-based assessment and periodic reviews."')
        print('  }')
        
        try:
            json_input = self._get_json_input()
            data = self._parse_json_input(json_input)
            
            action = data.get("action", "").strip()
            node_id = data.get("node_id", "").strip()
            new_details = data.get("new_details", "").strip()
            
            if action != "modify":
                print("❌ Action must be 'modify'.")
                return
            
            if not node_id or not new_details:
                print("❌ Both 'node_id' and 'new_details' fields are required.")
                return
            
            print(f"\n🔄 Modifying node: {node_id}")
            result = self.rag_system.modify_node(node_id, new_details)
            
            print("\n✅ RESULT:")
            print(json.dumps(result, indent=2))
            
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Error modifying node: {e}")
    
    def delete_node_json(self):
        print("\n🗑️ DELETE GRAPH NODE")
        print('Single-line: {"action": "delete", "node_id": "Policy_Section_5.1"}')
        print('Multi-line:')
        print('  {')
        print('    "action": "delete",')
        print('    "node_id": "Policy_Section_5.1"')
        print('  }')
        
        try:
            json_input = self._get_json_input()
            data = self._parse_json_input(json_input)
            
            action = data.get("action", "").strip()
            node_id = data.get("node_id", "").strip()
            
            if action != "delete":
                print("❌ Action must be 'delete'.")
                return
            
            if not node_id:
                print("❌ 'node_id' field is required.")
                return
            
            print(f"\n🔄 Deleting node: {node_id}")
            result = self.rag_system.delete_node(node_id)
            
            print("\n✅ RESULT:")
            print(json.dumps(result, indent=2))
            
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Error deleting node: {e}")
    
    def visualize_graph(self):
        try:
            print("\n📊 GRAPH VISUALIZATION")
            visualization = self.rag_system.visualize_current_graph()
            print(visualization)
        except Exception as e:
            print(f"❌ Error visualizing graph: {e}")
    
    def show_statistics(self):
        try:
            print("\n📈 GRAPH STATISTICS")
            stats = self.rag_system.get_graph_statistics()
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")

def main():
    try:
        cli = GraphRAGCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ System Error: {e}")

if __name__ == "__main__":
    main()
