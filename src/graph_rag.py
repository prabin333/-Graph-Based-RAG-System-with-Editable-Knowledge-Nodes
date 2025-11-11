import os
import json
from typing import Dict, List
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from entity_extractor import EntityExtractor
from graph_builder import KnowledgeGraphBuilder
from graph_visualizer import GraphVisualizer
from config import config
from pprint import pprint

class GraphRAGSystem:
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.visualizer = GraphVisualizer()
        self.current_graph_name = None
        self.llm_pipeline = None
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        try:
            if not config.LLM_MODEL_PATH:
                raise ValueError("LLM model path not configured")
            
            print("🔹 Loading LLM for answer generation...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("✅ LLM loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            self.llm_pipeline = None
    
    def process_document_with_query(self, input_data: Dict) -> Dict:
        document_path = input_data.get("document", "").strip()
        query = input_data.get("query", "").strip()
        
        if not document_path:
            return {
                "answer": "Document path is required",
                "graph_status": "Processing failed"
            }
        
        try:
            print("\n" + "="*60)
            print("🚀 STARTING DOCUMENT PROCESSING PIPELINE")
            print("="*60)
            
            print(f"📄 Processing document: {document_path}")
            
            print("\n📖 STEP 1: Reading document text...")
            text = self._extract_text_from_file(document_path)
            print(f"✅ Read {len(text)} characters from document")
            
            print("\n🔍 STEP 2: Entity and relationship extraction...")
            extraction_result = self.entity_extractor.extract_entities_relationships(text)
            
            print("\n🕸️ STEP 3: Building knowledge graph...")
            graph = self.graph_builder.build_from_extraction(extraction_result)
            print(f"✅ Built knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            doc_name = os.path.basename(document_path).split('.')[0]
            self.graph_builder.save_graph(doc_name)
            self.current_graph_name = doc_name
            
            print("\n📊 STEP 4: Graph visualization...")
            visualization = self.visualizer.visualize_graph_text(graph)
            print(visualization)
            
            if query:
                print(f"\n❓ STEP 5: Answering query: {query}")
                answer_result = self.query_graph(query)
                return {
                    "answer": answer_result["answer"],
                    "graph_status": f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
                }
            else:
                return {
                    "answer": f"Document processed successfully. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges",
                    "graph_status": "Document processed successfully"
                }
                
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return {
                "answer": f"Error processing document: {e}",
                "graph_status": "Processing failed"
            }
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    
    def query_graph(self, question: str) -> Dict:
        if not hasattr(self.graph_builder, 'graph') or self.graph_builder.graph.number_of_nodes() == 0:
            return {
                "answer": "No knowledge graph available. Please upload a document first."
            }
        
        if self.llm_pipeline is None:
            return {
                "answer": "LLM not available for answering questions."
            }
        
        print("🔍 Searching graph for relevant information...")
        relevant_nodes = self._find_relevant_nodes_in_graph(question)
        graph_context = self._extract_graph_context(relevant_nodes)
        
        print("🤖 Generating answer using graph context...")
        answer = self._generate_with_llm_using_graph(question, graph_context)
        
        return {
            "answer": answer
        }
    
    def _find_relevant_nodes_in_graph(self, question: str) -> List[str]:
        relevant_nodes = []
        graph = self.graph_builder.graph
        
        question_lower = question.lower()
        
        for node_id, node_data in graph.nodes(data=True):
            node_content = ""
            if "content" in node_data:
                node_content = node_data["content"].lower()
            elif "label" in node_data:
                node_content = node_data["label"].lower()
            elif "description" in node_data:
                node_content = node_data["description"].lower()
            
            keywords = ["compliance", "requirement", "policy", "data", "security", 
                       "encryption", "audit", "storage", "processing", "sharing",
                       "consent", "protection", "governance", "risk", "review"]
            
            for keyword in keywords:
                if keyword in question_lower and keyword in node_content:
                    if node_id not in relevant_nodes:
                        relevant_nodes.append(node_id)
                    break
        
        print(f"📍 Found {len(relevant_nodes)} relevant nodes in graph")
        return relevant_nodes
    
    def _extract_graph_context(self, node_ids: List[str]) -> str:
        graph = self.graph_builder.graph
        context_parts = []
        
        for node_id in node_ids[:8]:  
            if node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                
                node_info = f"NODE: {node_id}\n"
                if "label" in node_data:
                    node_info += f"Title: {node_data['label']}\n"
                if "type" in node_data:
                    node_info += f"Type: {node_data['type']}\n"
                if "content" in node_data:
                    content = node_data['content']
                    if len(content) > 300:
                        content = content[:300] + "..."
                    node_info += f"Content: {content}\n"
                if "description" in node_data:
                    node_info += f"Description: {node_data['description']}\n"
                
                predecessors = list(graph.predecessors(node_id))
                successors = list(graph.successors(node_id))
                
                if predecessors:
                    node_info += f"Connected from: {', '.join(predecessors[:3])}\n"
                if successors:
                    node_info += f"Connects to: {', '.join(successors[:3])}\n"
                
                context_parts.append(node_info)
        
        if not context_parts:
            return "No relevant nodes found in the knowledge graph."
        
        return "\n\n".join(context_parts)
    
    def _generate_with_llm_using_graph(self, question: str, graph_context: str) -> str:
        if not graph_context:
            return "I couldn't find relevant information in the knowledge graph to answer this question."
        
        prompt = f"""Based on the following knowledge graph structure, answer the question concisely and accurately.

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

QUESTION: {question}

Provide a clear answer based only on the information in the knowledge graph context.

ANSWER: """
        
        try:
            result = self.llm_pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                return_full_text=False
            )
            
            answer = result[0]["generated_text"].strip()
            return answer
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def modify_node(self, node_id: str, new_details: str) -> Dict:
        """Modify a node in the knowledge graph - returns structured system response"""
        if not node_id or not new_details:
            return {
                "answer": "Both node_id and new_details are required",
                "modified_node": node_id or "unknown",
                "graph_status": "Modification failed - missing parameters"
            }
        
        success = self.graph_builder.modify_node(node_id, new_details)
        
        if success:
            return {
                "answer": f"Node {node_id} has been successfully updated with the new details",
                "modified_node": node_id,
                "graph_status": "Node updated successfully and graph re-indexed"
            }
        else:
            return {
                "answer": f"Node {node_id} not found in the knowledge graph",
                "modified_node": node_id,
                "graph_status": "Node not found"
            }
    
    def delete_node(self, node_id: str) -> Dict:
        if not node_id:
            return {
                "answer": "node_id is required",
                "deleted_node": "unknown",
                "graph_status": "Deletion failed - missing node_id"
            }
        
        success = self.graph_builder.delete_node(node_id)
        
        if success:
            return {
                "answer": f"Node {node_id} has been successfully deleted from the knowledge graph",
                "deleted_node": node_id,
                "graph_status": "Node deleted successfully and graph re-indexed"
            }
        else:
            return {
                "answer": f"Node {node_id} not found in the knowledge graph",
                "deleted_node": node_id,
                "graph_status": "Node not found"
            }
    
    def visualize_current_graph(self) -> str:
        if hasattr(self.graph_builder, 'graph'):
            return self.visualizer.visualize_graph_text(self.graph_builder.graph)
        else:
            return "No graph loaded. Please process a document first."
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the current graph"""
        if hasattr(self.graph_builder, 'graph'):
            return self.visualizer.get_graph_stats(self.graph_builder.graph)
        else:
            return {"error": "No graph loaded"}
