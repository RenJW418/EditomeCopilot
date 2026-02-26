from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Any
from core.llm_client import LLMClient
import json
import re

class StructuredFilter(BaseModel):
    """
    Defines the structure for filtering gene editing literature.
    """
    technology: Optional[str] = Field(None, description="The specific gene editing technology (e.g., 'Base Editing', 'Prime Editing', 'CRISPR-Cas9').")
    min_efficiency: Optional[float] = Field(None, description="Minimum editing efficiency percentage (0-100).")
    max_off_target: Optional[float] = Field(None, description="Maximum acceptable off-target percentage (0-100).")
    cell_type: Optional[str] = Field(None, description="Specific cell type (e.g., 'HEK293', 'T-cell', 'HSC').")
    species: Optional[str] = Field(None, description="Target species (e.g., 'Human', 'Mouse', 'Plant').")
    delivery_system: Optional[str] = Field(None, description="Delivery method (e.g., 'AAV', 'LNP', 'Electroporation').")
    evidence_level: Optional[str] = Field(None, description="Minimum evidence level required (e.g., 'Clinical', 'Animal').")
    year_range: Optional[List[int]] = Field(None, description="A list of two integers representing the start and end year (e.g., [2020, 2024]).")

class SelfQueryRetriever:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_filters(self, query: str) -> dict:
        """
        Analyzes the user query to extract structured filters for the literature database.
        Returns a dictionary of filters to be applied during retrieval.
        """
        print(f"\n[Self-Querying] Analyzing query for structured filters: '{query}'")
        
        prompt = f"""
        You are an expert query analyzer for a scientific literature database.
        Your task is to extract specific filtering criteria from the user's natural language query.
        
        The database contains fields: 
        - technology (str)
        - efficiency (float, %)
        - off_target (float, %)
        - cell_type (str)
        - species (str)
        - delivery (str)
        - evidence_level (str)
        - year (int)

        User Query: "{query}"

        Instructions:
        1. Identify any explicit constraints (e.g., "efficiency > 50%", "in human cells", "published after 2020").
        2. Return a SINGLE JSON object representing the filters.
        3. Use "null" for fields not mentioned.
        4. For 'efficiency' and 'off_target', extract numeric values.
        5. For 'year', extract a range if applicable (e.g., "recent" -> [2023, 2026]).

        Example JSON Output:
        {{
            "technology": "Base Editing",
            "min_efficiency": 50.0,
            "max_off_target": null,
            "cell_type": "T-cell",
            "species": "Human",
            "delivery_system": null,
            "evidence_level": null,
            "year_start": 2020,
            "year_end": null
        }}
        
        Respond ONLY with the JSON object.
        """
        
        try:
            response = self.llm.generate(prompt, system_prompt="You are a precise query parser.", enable_thinking=False, max_tokens=300)
            filters = self._safe_parse_json(response)
            
            # Clean up nulls
            if filters:
                filters = {k: v for k, v in filters.items() if v is not None}
                print(f"[Self-Querying] Extracted filters: {filters}")
                return filters
            return {}
        except Exception as e:
            print(f"[Self-Querying] Filter extraction failed: {e}")
            return {}

    def _safe_parse_json(self, text):
        if not text: return None
        cleaned = str(text).replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except: pass
        return None

    def apply_filters(self, candidates: List[dict], filters: dict) -> List[dict]:
        """
        Applies the extracted filters to a list of candidate documents (metadata).
        """
        if not filters:
            return candidates
            
        filtered_results = []
        print(f"[Self-Querying] Filtering {len(candidates)} candidates...")
        
        for item in candidates:
            meta = item.get("metadata", {})
            struct = meta.get("structured_data", {}) or {}
            
            # --- Logic for each filter ---
            
            # 1. Technology (Substring match)
            if "technology" in filters:
                doc_tech = struct.get("technology", "Unknown")
                if filters["technology"].lower() not in doc_tech.lower() and doc_tech != "Unknown":
                    continue # Skip if tech doesn't match and isn't unknown

            # 2. Efficiency (Numeric comparison)
            if "min_efficiency" in filters:
                doc_eff_str = str(struct.get("efficiency", "0")).replace("%", "")
                try:
                    # Extract first number found
                    eff_val = float(re.findall(r"[\d\.]+", doc_eff_str)[0])
                    if eff_val < filters["min_efficiency"]:
                        continue
                except:
                    pass # Keep if efficiency is unparseable (don't be too strict)

            # 3. Cell Type (Substring)
            if "cell_type" in filters:
                doc_cell = struct.get("cell_type", "Unknown")
                if filters["cell_type"].lower() not in doc_cell.lower():
                    continue

            # 4. Species
            if "species" in filters:
                doc_species = struct.get("species", "Unknown")
                if filters["species"].lower() not in doc_species.lower():
                     continue

            # 5. Year Range
            if "year_start" in filters:
                try:
                    doc_year = int(meta.get("year", "0"))
                    if doc_year < filters["year_start"]:
                        continue
                except: pass
                
            if "year_end" in filters:
                try:
                     doc_year = int(meta.get("year", "0"))
                     if doc_year > filters["year_end"]:
                         continue
                except: pass

            filtered_results.append(item)
            
        print(f"[Self-Querying] Retained {len(filtered_results)} documents after filtering.")
        return filtered_results
