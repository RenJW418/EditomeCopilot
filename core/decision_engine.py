import json
import re
import os

class DecisionEngine:
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self.use_llm_parse = os.getenv("USE_LLM_PARSE", "false").lower() in ["1", "true", "yes"]

    def _heuristic_parse_query(self, query):
        q = (query or "").lower()

        if "point mutation" in q or " snv" in q or re.search(r"\b[acgt]\s*>\s*[acgt]\b", q):
            mutation_type = "SNV"
        elif "insertion" in q or "insert" in q:
            mutation_type = "insertion"
        elif "deletion" in q or "delete" in q:
            mutation_type = "deletion"
        else:
            mutation_type = "Unknown"

        if "hsc" in q:
            cell_type = "HSC"
        elif "hek293" in q:
            cell_type = "HEK293"
        elif "t-cell" in q or "t cell" in q or "tcell" in q:
            cell_type = "T-cell"
        else:
            cell_type = "Unknown"

        if "human" in q:
            species = "human"
        elif "mouse" in q or "murine" in q:
            species = "mouse"
        else:
            species = "Unknown"

        if "in vivo" in q:
            delivery_constraint = "in vivo"
        elif "in vitro" in q:
            delivery_constraint = "in vitro"
        elif "aav" in q:
            delivery_constraint = "AAV"
        elif "lnp" in q:
            delivery_constraint = "LNP"
        elif "electroporation" in q:
            delivery_constraint = "electroporation"
        else:
            delivery_constraint = "Unknown"

        return {
            "mutation_type": mutation_type,
            "cell_type": cell_type,
            "species": species,
            "delivery_constraint": delivery_constraint,
        }

    def parse_query(self, query, llm_client=None):
        """Use LLM to parse the user query into structured constraints."""
        if (not llm_client) or (not self.use_llm_parse):
            return self._heuristic_parse_query(query)

        prompt = f"""
        Extract the following constraints from the user query for gene editing.
        Return ONLY a valid JSON object with keys:
        - "mutation_type": (e.g., SNV, insertion, deletion)
        - "cell_type": (e.g., HSC, HEK293)
        - "species": (e.g., human, mouse)
        - "delivery_constraint": (e.g., in vivo, in vitro, AAV, LNP)

        If a constraint is not mentioned, use "Unknown".

        Query: {query}
        """
        try:
            response = llm_client.generate(
                prompt,
                system_prompt="You are a precise constraint extraction agent. Output only JSON.",
                enable_thinking=False,
                timeout=15,
                max_tokens=220,
            )
            if not response or str(response).startswith("Error calling LLM API"):
                raise ValueError(response or "Empty LLM response")

            parsed = self._safe_parse_json(response)
            if parsed and isinstance(parsed, dict):
                return {
                    "mutation_type": parsed.get("mutation_type", "Unknown"),
                    "cell_type": parsed.get("cell_type", "Unknown"),
                    "species": parsed.get("species", "Unknown"),
                    "delivery_constraint": parsed.get("delivery_constraint", "Unknown"),
                }
            raise ValueError("LLM constraint response is not valid JSON")
        except Exception as e:
            print(f"Error parsing query: {e}")
            return self._heuristic_parse_query(query)

    def _safe_parse_json(self, text):
        if not text:
            return None

        cleaned = str(text).replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def calculate_score(self, tech_data, constraints):
        """
        Calculate the recommendation score based on the formula:
        score = efficiency_weight - off_target_weight + evidence_weight + delivery_compatibility
        """
        score = 0.0
        details = {}
        
        # 1. Efficiency Weight (0 to 40 points)
        efficiency_str = tech_data.get("efficiency", "0%")
        try:
            # Extract numeric value from string like "65%" or "up to 80%"
            import re
            nums = re.findall(r'\d+', efficiency_str)
            if nums:
                eff_val = float(nums[0])
                eff_score = (eff_val / 100.0) * 40
            else:
                eff_score = 10 # Default low score if unknown
        except:
            eff_score = 10
        score += eff_score
        details["efficiency_score"] = round(eff_score, 2)

        # 2. Off-target Weight (Penalty: 0 to -30 points)
        off_target = tech_data.get("off_target_risk", "Unknown").lower()
        if "low" in off_target or "none" in off_target:
            ot_penalty = 0
        elif "medium" in off_target:
            ot_penalty = -15
        elif "high" in off_target:
            ot_penalty = -30
        else:
            ot_penalty = -10 # Unknown risk penalty
        score += ot_penalty
        details["off_target_penalty"] = ot_penalty

        # 3. Evidence Weight (0 to 20 points)
        evidence = tech_data.get("evidence_level", "Unknown")
        if "Level 1" in evidence: # Clinical
            ev_score = 20
        elif "Level 2" in evidence: # Animal
            ev_score = 15
        elif "Level 3" in evidence: # In vitro
            ev_score = 10
        elif "Level 4" in evidence: # Predictive
            ev_score = 5
        else:
            ev_score = 0
        score += ev_score
        details["evidence_score"] = ev_score

        # 4. Delivery Compatibility (0 to 10 points)
        # Simplified logic: if the study used a delivery method compatible with the constraint
        delivery_used = tech_data.get("delivery_system", "Unknown").lower()
        constraint_del = constraints.get("delivery_constraint", "Unknown").lower()
        
        if constraint_del != "unknown" and constraint_del != "unknown":
            if constraint_del in delivery_used or delivery_used in constraint_del:
                del_score = 10
            elif "in vivo" in constraint_del and ("aav" in delivery_used or "lnp" in delivery_used):
                del_score = 10
            else:
                del_score = 0
        else:
            del_score = 5 # Neutral if unknown
            
        score += del_score
        details["delivery_score"] = del_score
        
        # 5. Context Match Bonus (Cell Type / Species)
        if constraints.get("cell_type", "Unknown").lower() != "unknown" and constraints["cell_type"].lower() in tech_data.get("cell_type", "").lower():
            score += 10
            details["context_bonus"] = 10
            
        return round(score, 2), details

    def evaluate(self, query, llm_client=None):
        constraints = self.parse_query(query, llm_client)
        
        # If we have a KG, we would query it here.
        # For demonstration, we'll mock some retrieved data from the KG based on the mutation type.
        
        mock_kg_results = []
        mut_type = constraints.get("mutation_type", "").upper()
        
        if "SNV" in mut_type or "POINT" in mut_type:
            mock_kg_results.extend([
                {
                    "technology": "Base Editing",
                    "efficiency": "75%",
                    "off_target_risk": "Low",
                    "evidence_level": "Level 2",
                    "delivery_system": "AAV",
                    "cell_type": "HSC"
                },
                {
                    "technology": "Prime Editing",
                    "efficiency": "45%",
                    "off_target_risk": "Low",
                    "evidence_level": "Level 3",
                    "delivery_system": "LNP",
                    "cell_type": "HEK293"
                }
            ])
        else:
            mock_kg_results.extend([
                {
                    "technology": "CRISPR KO",
                    "efficiency": "85%",
                    "off_target_risk": "Medium",
                    "evidence_level": "Level 1",
                    "delivery_system": "AAV",
                    "cell_type": "T-cells"
                }
            ])

        # Score each technology
        scored_results = []
        for tech_data in mock_kg_results:
            score, details = self.calculate_score(tech_data, constraints)
            scored_results.append({
                "technology": tech_data["technology"],
                "expected_efficiency": tech_data["efficiency"],
                "off_target_risk": tech_data["off_target_risk"],
                "score": score,
                "scoring_details": details,
                "raw_data": tech_data
            })
            
        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Format output
        report = f"Query: {query}\n"
        report += f"Parsed Constraints: {json.dumps(constraints, indent=2)}\n\n"
        report += "Decision Engine Recommendations:\n"
        report += "-" * 50 + "\n"
        
        for res in scored_results:
            report += f"Technology: {res['technology']}\n"
            report += f"  Score: {res['score']}/100\n"
            report += f"  Expected Efficiency: {res['expected_efficiency']}\n"
            report += f"  Off-target Risk: {res['off_target_risk']}\n"
            report += f"  Scoring Details: {json.dumps(res['scoring_details'])}\n"
            report += "-" * 50 + "\n"
            
        return report

if __name__ == "__main__":
    engine = DecisionEngine()
    print(engine.evaluate("Fix a G>A SNV mutation in HSC cells for in vivo delivery"))
