import os
import re
import json
import time
import torch
from typing import Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import config

class EntityExtractor:
    def __init__(self):
        self.llm_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_llm()
    
    def _initialize_llm(self):
        try:
            if not config.LLM_MODEL_PATH:
                raise ValueError("LLM model path not configured")
            
            print(f"🔹 Loading Gemma for entity extraction on {self.device.upper()}...")

            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True
            }
            
            if self.device == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                })
            
            model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_PATH,
                **model_kwargs
            )
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
            }
            
            if self.device == "cuda":
                pipeline_kwargs["torch_dtype"] = torch.float16
                pipeline_kwargs["device_map"] = "auto"
            
            self.llm_pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )
            
            print(f"✅ Gemma loaded successfully on {self.device.upper()}!")
            
        except Exception as e:
            print(f"❌ Failed to load Gemma for entity extraction: {e}")
            self.llm_pipeline = None
    
    def extract_entities_relationships(self, text: str) -> Dict:
        if not self.llm_pipeline:
            print("❌ LLM not available, returning empty extraction")
            return {
                "entities": [],
                "relationships": [], 
                "policy_sections": []
            }
        
        print("🔄 Starting LLM extraction process...")
        start_time = time.time()
        
        try:
            extraction_result = self._extract_everything_with_llm(text)
            end_time = time.time()
            
            print(f"✅ LLM extraction completed in {end_time - start_time:.2f} seconds")
            print("📊 EXTRACTION RESULTS SUMMARY:")
            print(f"   - Sections: {len(extraction_result.get('policy_sections', []))}")
            print(f"   - Entities: {len(extraction_result.get('entities', []))}")
            print(f"   - Relationships: {len(extraction_result.get('relationships', []))}")
            
            return extraction_result
            
        except Exception as e:
            print(f"❌ LLM extraction failed: {e}")
            return {
                "entities": [],
                "relationships": [], 
                "policy_sections": []
            }
    
    def _extract_everything_with_llm(self, text: str) -> Dict:
        try:
            print("📝 Preparing document for LLM analysis...")
            
            # INCREASED CHARACTER LIMIT to capture full document
            truncated_text = text[:4000] if len(text) > 4000 else text
            print(f"📄 Using first {len(truncated_text)} characters for analysis")
            
            # IMPROVED PROMPT with better guidance for compliance requirements
            prompt = f"""Analyze this policy document and extract ALL compliance requirements, entities, and relationships.

DOCUMENT:
{truncated_text}

Return ONLY this JSON format with ALL sections and requirements:
{{
  "policy_sections": [
    {{
      "id": "section_1",
      "title": "Section Title",
      "content": "Section content summary",
      "requirements": [
        {{
          "id": "req_1.1",
          "text": "Full requirement text",
          "full_reference": "Section 1.1",
          "compliance_type": "data_storage|encryption|processing|sharing|audit"
        }}
      ]
    }}
  ],
  "entities": [
    {{
      "id": "entity_1",
      "text": "Entity Name", 
      "type": "DATA|SERVER|PERSON|SYSTEM|PROCESS",
      "description": "Entity description"
    }}
  ],
  "relationships": [
    {{
      "from": "source_id",
      "to": "target_id", 
      "relationship": "relationship_type",
      "text": "Relationship description"
    }}
  ]
}}

IMPORTANT: Extract ALL requirements from ALL sections. Focus on compliance requirements.

Return ONLY the JSON, no other text:"""

            print("🤖 Sending to LLM for analysis...")
            
            generation_kwargs = {
                "max_new_tokens": 1200,  
                "temperature": 0.1,
                "do_sample": False,
                "return_full_text": False,
                "pad_token_id": self.llm_pipeline.tokenizer.eos_token_id,
            }
            
            result = self.llm_pipeline(prompt, **generation_kwargs)
            
            response_text = result[0]["generated_text"].strip()
            print(f"📨 LLM response received ({len(response_text)} characters)")
            
            json_data = self._extract_and_validate_json(response_text)
            
            return self._ensure_extraction_structure(json_data)
                
        except Exception as e:
            print(f"❌ LLM extraction failed: {e}")
            return {
                "entities": [],
                "relationships": [],
                "policy_sections": []
            }
    
    def _extract_and_validate_json(self, response_text: str) -> Dict:
        """Extract and validate JSON with detailed debugging"""
        try:
            print("🔄 Starting JSON extraction...")
            
            json_str = self._extract_first_json_object(response_text)
            
            if not json_str:
                print("❌ No JSON string extracted")
                return {}
            
            print(f"📄 Extracted JSON string ({len(json_str)} chars)")
            
            for attempt in range(3):
                try:
                    data = json.loads(json_str)
                    print(f"✅ JSON parsed successfully on attempt {attempt + 1}")
                    return data
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing attempt {attempt + 1} failed: {e}")
                    
                    if attempt < 2:
                        print("🛠️  Attempting to fix JSON issues...")
                        json_str = self._fix_json_issues(json_str)
            
            print("❌ All JSON parsing attempts failed")
            return {}
            
        except Exception as e:
            print(f"❌ JSON extraction error: {e}")
            return {}
    
    def _extract_first_json_object(self, response_text: str) -> str:
        """Extract the first complete JSON object from the response"""
        cleaned_text = re.sub(r'```json\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        depth = 0
        json_start = -1
        json_chars = []
        
        for i, char in enumerate(cleaned_text):
            if char == '{':
                if depth == 0:
                    json_start = i
                    json_chars = []
                depth += 1
                json_chars.append(char)
            elif char == '}':
                depth -= 1
                json_chars.append(char)
                if depth == 0 and json_start != -1:
                    return ''.join(json_chars)
            elif json_start != -1:
                json_chars.append(char)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text)
        if json_match:
            return json_match.group()
        brace_start = cleaned_text.find('{')
        brace_end = cleaned_text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return cleaned_text[brace_start:brace_end+1]
        
        return ""
    
    def _fix_json_issues(self, json_str: str) -> str:
        fixes = [
            (r',\s*\}', '}'),
            (r',\s*\]', ']'),
            (r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
            (r':\s*\'([^\']+)\'', r': "\1"'),
            (r'\'([^\']+)\'\s*:', r'"\1":'),
            (r'//.*', ''),
            (r'}\s*{', '}, {'),
            (r']\s*{', '], {'),
            (r'^[^{]*', ''),
            (r'[^}]*$', ''),
        ]
        
        fixed = json_str
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, fixed, flags=re.MULTILINE | re.DOTALL)
        
        return fixed
    
    def _ensure_extraction_structure(self, data: Dict) -> Dict:
        """Ensure the extraction result has the proper structure"""
        if not isinstance(data, dict):
            return {
                "entities": [],
                "relationships": [],
                "policy_sections": []
            }
        
        sections = data.get("policy_sections", data.get("sections", []))
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        print(f"📊 Raw extraction counts: {len(sections)} sections, {len(entities)} entities, {len(relationships)} relationships")
        
        for i, section in enumerate(sections):
            if "id" not in section:
                section["id"] = f"section_{i+1}"
            if "content" not in section:
                section["content"] = section.get("title", "")
            if "requirements" not in section:
                section["requirements"] = []
            
            for j, req in enumerate(section.get("requirements", [])):
                if "id" not in req:
                    req["id"] = f"req_{i+1}.{j+1}"
                if "full_reference" not in req:
                    req["full_reference"] = f"Section {i+1}.{j+1}"
        
        for rel in relationships:
            if "text" not in rel:
                rel["text"] = rel.get("description", f"Relationship {rel.get('from')} to {rel.get('to')}")
        
        for entity in entities:
            if "sentence" not in entity:
                entity["sentence"] = entity.get("description", "")
        
        result = {
            "entities": entities,
            "relationships": relationships,
            "policy_sections": sections
        }
        
        print(f"✅ Final structured extraction: {len(sections)} sections, {len(entities)} entities, {len(relationships)} relationships")
        return result