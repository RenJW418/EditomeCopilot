from core.knowledge_graph import GeneEditingKnowledgeGraph
from core.decision_engine import DecisionEngine
from core.risk_assessment import RiskAssessor
from core.llm_client import LLMClient
from core.data_pipeline import GeneEditingDataPipeline
from core.self_query import SelfQueryRetriever
from core.graph_reasoning import GraphReasoner
from core.review_agent import ReviewGenerator
import json
import re
import os

class AgenticRAG:
    def __init__(self):
        self.kg = GeneEditingKnowledgeGraph()
        self.llm = LLMClient()
        self.use_llm_routing = os.getenv("USE_LLM_ROUTING", "false").lower() in ["1", "true", "yes"]
        
        # Initialize the 9-step Data Pipeline for Retrieval
        print("Initializing Gene Editing Data Pipeline (FAISS + HuggingFace Embeddings)...")
        self.data_pipeline = GeneEditingDataPipeline(base_dir="data")
        
        self.decision_engine = DecisionEngine(knowledge_graph=self.kg)
        self.risk_assessor = RiskAssessor()
        
        # New Capabilities
        self.self_query_retriever = SelfQueryRetriever(self.llm)
        self.graph_reasoner = GraphReasoner(self.kg.graph)
        self.review_agent = ReviewGenerator(self.llm, self.data_pipeline)

    def _detect_language(self, text):
        t = text or ""
        # Chinese if contains CJK Unified Ideographs
        if re.search(r"[\u4e00-\u9fff]", t):
            return "zh"
        return "en"

    def _labels(self, lang):
        if lang == "zh":
            return {
                "title": "🧬 决策增强生成（DAG）报告",
                "query": "问题",
                "literature": "📚 1. 文献证据（溯源）",
                "literature_empty": "本地索引中未检索到高相关证据。系统可能仍在自动更新，请稍后重试。",
                "kg": "🔗 2. 知识图谱（能力）",
                "decision": "⚙️ 3. 决策引擎（量化评分）",
                "risk": "⚠️ 4. 风险评估",
                "synthesis": "🤖 5. AI 综合结论",
                "disabled": "LLM 综合功能当前不可用，已返回可复现的 DAG 结构化结果。",
                "evidence_summary": "证据摘要",
                "raw_chunk": "原文片段",
            }
        return {
            "title": "🧬 Decision-Augmented Generation (DAG) Report",
            "query": "Query",
            "literature": "📚 1. Literature Evidence (Provenance)",
            "literature_empty": "No highly relevant local evidence found. Auto-update may still be running; please retry shortly.",
            "kg": "🔗 2. Knowledge Graph (Capabilities)",
            "decision": "⚙️ 3. Decision Engine (Quantitative Scoring)",
            "risk": "⚠️ 4. Risk Assessment",
            "synthesis": "🤖 5. AI Synthesis",
            "disabled": "LLM synthesis is currently unavailable. Returning deterministic DAG outputs.",
            "evidence_summary": "Evidence Summary",
            "raw_chunk": "Raw Chunk",
        }

    def _heuristic_intent(self, query):
        q = (query or "").lower()
        # Add Chinese keywords for better heuristic matching
        user_library_keywords = ["upload", "import", "my paper", "my lit", "these", "this", "上传", "导入", "我的文献", "刚上传"]
        return {
            "needs_retrieval": True,
            "needs_kg": any(k in q for k in ["crispr", "cas9", "base editing", "prime editing", "rna editing"]),
            "needs_decision": any(k in q for k in ["fix", "mutation", "recommend", "strategy", "point mutation", "snv"]),
            "needs_risk": any(k in q for k in ["risk", "safe", "safety", "off-target", "off target"]),
            "search_scope": "user_library" if any(k in q for k in user_library_keywords) else "general_database"
        }

    def analyze_intent(self, query):
        """
        Uses LLM to dynamically determine which modules to call based on user query.
        """
        if (not self.llm.client) or (not self.use_llm_routing):
            return self._heuristic_intent(query)

        prompt = f"""
        Analyze the following user query about gene editing or precision oncology.
        Determine which modules of the system need to be executed to answer the query.
        
        Modules:
        - needs_retrieval: True if the user is asking for literature, facts, or general knowledge.
        - needs_kg: True if the user is asking about specific technologies, capabilities, or known relationships.
        - needs_decision: True if the user is asking for a recommendation on WHICH editing technology to use for a specific mutation or context.
        - needs_risk: True if the user is asking about safety, off-target effects, or risks of a specific technology.
        - search_scope: 'user_library' if the user explicitly refers to their uploaded or imported files (e.g., 'my upload', 'imported paper', 'these files', 'what did I just upload', 'summary of my papers', '我上传', '导入的', '我的文献'), otherwise 'general_database'.
        - target_author: Extract the name of the specific author if the user asks about a specific author's work (e.g., "Zhang Feng", "Jennifer Doudna"). If no specific author, return null.

        Query: "{query}"
        
        Return ONLY a valid JSON object.
        Example: {{"needs_retrieval": true, "needs_kg": false, "needs_decision": true, "needs_risk": false, "search_scope": "general_database", "target_author": "Zhang Feng"}}
        """
        
        try:
            response = self.llm.generate(
                prompt,
                system_prompt="You are a JSON-only routing agent.",
                enable_thinking=False, # Disable thinking for faster routing
                timeout=15,
                max_tokens=200,
            )
            if not response or str(response).startswith("Error calling LLM API"):
                # Non-critical failure, fallback to heuristic
                print(f"LLM Routing failed: {response}. Using heuristic.")
                raise ValueError(response or "Empty LLM response")
            
            intent = self._safe_parse_json(response)
            
            # Fallback if JSON parsing fails even after cleaning
            if not intent:
                 raise ValueError("Parsed intent is None")

            # Check if heuristic strongly suggests user library even if LLM missed it
            heuristic = self._heuristic_intent(query)
            if intent.get("search_scope") != "user_library" and heuristic["search_scope"] == "user_library":
                intent["search_scope"] = "user_library"
                print("LLM routing overridden by heuristic for user library intent.")

            if intent and all(k in intent for k in ["needs_retrieval", "needs_kg", "needs_decision", "needs_risk"]):
                return {
                    "needs_retrieval": bool(intent.get("needs_retrieval")),
                    "needs_kg": bool(intent.get("needs_kg")),
                    "needs_decision": bool(intent.get("needs_decision")),
                    "needs_risk": bool(intent.get("needs_risk")),
                    "search_scope": intent.get("search_scope", "general_database"),
                    "target_author": intent.get("target_author")
                }
            raise ValueError("LLM routing response is not valid JSON intent")
        except Exception as e:
            print(f"Intent analysis failed: {e}. Falling back to default routing.")
            return self._heuristic_intent(query)

    def _safe_parse_json(self, text):
        """Best-effort JSON extraction for LLM outputs that may include wrappers or extra text."""
        if not text:
            return None

        cleaned = str(text).replace("```json", "").replace("```", "").strip()

        # 1) Direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 2) Extract first JSON object block
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def process_query(self, query):
        print(f"\n--- Processing Query: '{query}' ---")
        lang = self._detect_language(query)
        L = self._labels(lang)
        
        # Step 0: Intent Analysis (Dynamic Routing)
        print("\n[DAG Step 0] Analyzing Intent...")
        intent = self.analyze_intent(query)
        print(f"Routing Plan: {intent}")
        
        # Cross-lingual Retrieval Support
        search_query = query
        if lang == "zh" and intent.get("needs_retrieval"):
            try:
                print("Translating query to English for better retrieval...")
                translation_prompt = f"Translate the following biomedical query to English. Output ONLY the translation. Query: {query}"
                translated = self.llm.generate(translation_prompt, system_prompt="You are a translator.", enable_thinking=False, max_tokens=100)
                if translated and not translated.startswith("Error"):
                    search_query = translated.strip()
                    print(f"Translated query: '{search_query}'")
            except Exception as e:
                print(f"Translation failed: {e}")

        # We will build the context string for the LLM, but NOT for the final user output yet
        user_facing_response = ""
        dag_context = f"Query: {query}\n\n"
        
        # Track sources for the footer
        citations = []

        # Step 1: Hybrid Retrieval (Using the new Data Pipeline)
        if intent.get("needs_retrieval"):
            # Increased retrieval depth for broad summarization
            # Target: As comprehensive as possible within LLM context window (e.g., 32k or 128k)
            # Increase TOP-K significantly to capture broad "summary" queries
            top_k_retrieval = 150 
            max_provenance = 60 # Feed up to 60 chunks to the LLM (approx 20k-30k tokens)
            
            # --- STRUCTURED FILTERING (NEW) ---
            # Extract filters (e.g., "efficiency > 50%")
            structured_filters = {}
            if intent.get("search_scope") != "user_library": # Only for general search usually
                 structured_filters = self.self_query_retriever.generate_filters(query)
                 if structured_filters:
                     print(f"Applying Filters: {structured_filters}")
            
            # --- USER UPLOAD FILTERING ---
            # Check if user explicitly asks about their uploaded content
            # We use the LLM-determined search scope, with a heuristic fallback
            is_user_scope = intent.get("search_scope") == "user_library"
            
            if not is_user_scope:
                 user_upload_keywords = [
                     "我上传", "我刚刚上传", "my upload", "uploaded file", "imported", "导入", 
                     "uploaded", "uploads", "upload", "these papers",
                     "我的文献", "我的库", "文献库", "my library", "my literature", "in my database"
                 ]
                 if any(k in query.lower() for k in user_upload_keywords):
                     is_user_scope = True

            if is_user_scope:
                print("User intent indicates search in UPLOADED LIBRARY.")
                
                # Intelligent Query Adjustment:
                # If user asks "What did I upload?" or "List my papers", we use a generic query to hit all docs.
                # If user asks "Does my upload contain X?", we keep 'X' as the search query.
                is_listing_request = any(k in query.lower() for k in ["有哪些", "list", "what papers", "what literature", "show me"])
                is_specific_search = any(k in query.lower() for k in ["关于", "about", "contain", "mention", "refer"])
                
                if is_listing_request and not is_specific_search:
                     # Use broad terms common in almost all scientific papers to ensure retrieval
                     search_query = "study analysis research method result conclusion"
                     print(f"Using generic query for upload retrieval: '{search_query}'")
                
                # If filtering, we might not need as many chunks if user just uploaded a few
                top_k_retrieval = 100
            
            # --- AUTHOR AWARE RETRIEVAL ---
            target_author = intent.get("target_author")
            if target_author:
                 print(f"Target Author Detected: {target_author}")
                 # Boost top_k for author search to ensure we recall their papers even if score is lower
                 top_k_retrieval = 500 
            
            print(f"\\n[DAG Step 1] Retrieving Context (FAISS Semantic Search) for: '{search_query}'...")
            try:
                # Retrieve from appropriate source
                if is_user_scope:
                    results = self.data_pipeline.step7_retrieve(search_query, top_k=top_k_retrieval, source="user")
                    
                    if not results:
                        print("No user-imported documents found. Returning empty result set (strict filtering).")
                        # Do NOT fall back to general database if user asked for specific library
                else:
                     results = self.data_pipeline.step7_retrieve(search_query, top_k=top_k_retrieval, source="main")
                     
                     # FALLBACK: If main DB yields no results for a specific query, check User DB automatically
                     if not results:
                         print("Main database yielded no results. Checking User Uploads as fallback...")
                         user_results = self.data_pipeline.step7_retrieve(search_query, top_k=top_k_retrieval, source="user")
                         if user_results:
                             print(f"Fallback successful! Found {len(user_results)} docs in user library.")
                             results = user_results
                     
                provenance = self.data_pipeline.step8_provenance(results, max_items=max_provenance)
            except Exception as e:
                print(f"Retrieval error: {e}. Falling back to unfiltered search.")
                results = self.data_pipeline.step7_retrieve(search_query, top_k=top_k_retrieval)
                provenance = self.data_pipeline.step8_provenance(results, max_items=max_provenance)

            # --- POST-RETRIEVAL FILTERING (STRUCTURED FILTERS) ---
            if structured_filters and provenance:
                 print("Applying structured filters to candidates...")
                 # Step 8 gives us a list of dicts with 'structured_data' inside.
                 # The self_query_retriever needs raw metadata or formatted results?
                 # It handles our formatted results if they contain the metadata.
                 # We added 'structured_data' to the step8 output in data_pipeline.py, so this works.
                 
                 provenance = self.self_query_retriever.apply_filters(provenance, structured_filters)
                 print(f"Candidates remaining after structured filtering: {len(provenance)}")

            # --- POST-RETRIEVAL FILTERING (AUTHOR) ---
            if target_author and provenance:
                print(f"Filtering results for author: {target_author}")
                author_filtered_provenance = []
                # Split target author into parts (e.g. "Zhang Feng" -> ["Zhang", "Feng"])
                author_parts = [p.lower() for p in target_author.split() if len(p) > 1]
                
                for p in provenance:
                    # Check metadata authors or evidence string
                    content_to_check = (p.get('evidence', '') + " " + p.get('text', '')[:500]).lower()
                    
                    matches = sum(1 for part in author_parts if part in content_to_check)
                    
                    is_match = False
                    if len(author_parts) > 1:
                         # Strict: Require nearly all parts. Allow 1 miss ONLY if > 2 parts.
                         if matches >= len(author_parts) - (1 if len(author_parts) > 2 else 0):
                             is_match = True
                    elif len(author_parts) == 1:
                         # Strict: Single name must be present
                         if matches >= 1:
                             is_match = True
                             
                    if is_match:
                        author_filtered_provenance.append(p)
                
                if author_filtered_provenance:
                    print(f"Found {len(author_filtered_provenance)} papers matching author '{target_author}'.")
                    provenance = author_filtered_provenance
                    provenance = provenance[:max_provenance]
                else:
                    print(f"Warning: No papers passed strict author filter for '{target_author}'. Reverting to semantic top matches.")

            # --- POST-RETRIEVAL RE-RANKING (KEYWORD BOOST) ---
            # To reduce "noise" (e.g., "leap2" vs "LEAPER"), we boost chunks that contain exact conceptual matches from the query.
            # Identify potential acronyms in the search query (all caps words of len >= 3)
            import re
            acronyms = [w for w in re.findall(r'\\b[A-Z]{3,}\\b', search_query)]
            
            if acronyms:
                print(f"Applying keyword boosting for acronyms: {acronyms}")
                boosted_provenance = []
                regular_provenance = []
                
                for p in provenance:
                    # Check if ANY acronym is present in the text or title (evidence)
                    text_content = (p.get('text', '') + " " + p.get('evidence', '')).upper()
                    if any(acr in text_content for acr in acronyms):
                        boosted_provenance.append(p)
                    else:
                        regular_provenance.append(p)
                
                # Re-assemble: Boosted first, then regular
                provenance = boosted_provenance + regular_provenance
                print(f"Boosted {len(boosted_provenance)} chunks containing specific acronyms.")

            print(f"Retrieved {len(provenance)} relevant chunks for synthesis.")
            
            # --- AGENTIC REVIEW GENERATION MODE ---
            # If the user asks for a "Review" or "Overview" explicitly, delegate to ReviewGenerator
            # Use broader keywords including Chinese
            is_review_request = any(k in query.lower() for k in ["review", "survey", "overview", "summary of progress", "综述", "总结", "review paper", "summary report"])
            
            # Only switch if we have enough material (>5 papers) and LLM is active
            if is_review_request and len(provenance) >= 3 and self.llm.client:
                print("[Switching to Review Agent mode]")
                # We use the review agent to generate the main body
                user_facing_response = self.review_agent.generate_review(query, provenance)
                
                # Append standard footer (Citations)
                citations_footer = "\n\n---\n**📚 Key Reference Sources (Top 15):**\n"
                for i, p in enumerate(provenance[:15], start=1):
                     citations_footer += f"{i}. **{p['evidence']}** (Relevance: {round(float(p.get('score', 0)), 2)})\n"
                
                return user_facing_response + citations_footer

            if provenance:
                dag_context += "### Literature Evidence (Original English Sources):\\n"
                for i, p in enumerate(provenance, start=1):
                    # Include structured data if available
                    struct_data = p.get("structured_data", {})
                    struct_str = f" [Extracted: {struct_data.get('technology', 'Unknown')} | Eff: {struct_data.get('efficiency', 'Unknown')}]" if struct_data else ""
                    
                    # Add to LLM Context
                    dag_context += f"{i}) [{p['evidence']}] (score={round(float(p.get('score', 0)), 4)}){struct_str}\\n{p['text']}\\n\\n"
                    
                    # Add to Citations Visual Object
                    citations.append(f"{i}. **{p['evidence']}** (Relevance: {round(float(p.get('score', 0)), 2)})")
                print(f"Retrieved {len(provenance)} chunks.")
            else:
                print("No relevant literature found in local FAISS index.")
                dag_context += "### Literature Evidence:\\nNo relevant literature found in local database.\\n\\n"

        # Step 2: Knowledge Graph Query
        if intent.get("needs_kg"):
            print("\\n[DAG Step 2] Querying Knowledge Graph...")
            techs = ["CRISPR KO", "Base Editing", "Prime Editing", "RNA editing", "Cas9", "Cas12a"]
            found_tech = None
            for t in techs:
                if t.lower() in query.lower():
                    found_tech = t
                    break
            
            if found_tech:
                caps = self.kg.query_technology_capabilities(found_tech)
                if caps:
                    print(f"Found capabilities for {found_tech}.")
                    kg_info = f"Technology: {found_tech}\\nCan Fix: {', '.join(caps['can_fix'])}\\nRecorded Studies: {len(caps['studies'])}"
                    dag_context += f"### Knowledge Graph Data:\\n{kg_info}\\n\\n"
                else:
                    print(f"No capabilities found for {found_tech} in KG.")
            else:
                print("No specific technology found in query for KG lookup.")

        # Step 3: Decision Engine (Quantitative Scoring)
        top_strategy = "CRISPR KO"
        if intent.get("needs_decision"):
            print("\\n[DAG Step 3] Quantitative Decision Engine...")
            decision_report = self.decision_engine.evaluate(query, self.llm)
            print("Strategies evaluated and scored.")
            dag_context += f"### Decision Engine Analysis:\\n{decision_report}\\n\\n"
            
            if "Technology: Base Editing" in decision_report and "Score: " in decision_report:
                pass

        # Step 4: Risk Assessment
        if intent.get("needs_risk"):
            print("\\n[DAG Step 4] Assessing Risks...")
            sequence = "ATGCGTACGTAGCTAG" # Dummy
            locus = "Target_Locus" # Dummy
            risk_report = self.risk_assessor.assess_risk(sequence, locus, top_strategy)
            print(f"Risk Assessment completed for {top_strategy}.")
            
            dag_context += f"### Risk Assessment ({top_strategy}):\\n"
            dag_context += f"- Overall Risk Level: {risk_report['risk_level']}\\n"
            dag_context += f"- Off-target Probability: {risk_report['off_target_probability']}\\n"
            dag_context += f"- Functional Disruption: {risk_report['functional_disruption_probability']}\\n"
            dag_context += f"- Uncertainty: {risk_report['uncertainty_interval']}\\n\\n"

        # Step 5: LLM Synthesis (Decision-Augmented Generation)
        if self.llm.client:
            print("\\n[DAG Step 5] Synthesizing Final Answer with LLM...")
            
            # --- UPDATED PROMPT FOR COMPREHENSIVE SUMMARIES ---
            synthesis_prompt = f"""
            You are a Decision-grade Gene Editing Almanac AI.
            Your task is to provide a comprehensive, structured, and highly readable summary based on the retrieved literature.
            
            IMPORTANT: The user wants a "Summary of Progress". Do not just list papers. Synthesize them into a coherent narrative.
            
            Guidelines for Output:
            1. **Structure**: Use clear Markdown headers (##), bullet points, and bold text for key concepts.
            2. **Readability**: Write in a smooth, professional, yet accessible style (like a high-quality review paper). 
            3. **Comprehensiveness**: Integrate ALL relevant retrieved evidence. Do not ignore details just because they are numerous. Group them.
            4. **Categorization**: Group findings by themes (e.g., "Delivery Systems", "Off-target Optimization", "Clinical Applications", "New Cas Variants") if applicable.
            5. **Citations**: When mentioning a specific finding, refer to the evidence index [1], [2] etc. naturally.
            6. **Language**: Respond in the SAME language as the query (Chinese for Chinese query).

            System Data (DAG Context / Retrieved Literature):
            {dag_context}
            
            User Query: {query}
            
            Begin your comprehensive report:
            """
            
            # Increase output token limit for longer summaries
            # Dynamic max_tokens based on content length
            max_output_tokens = 3000 if len(provenance) > 10 else 1500
            
            llm_summary = self.llm.generate(
                synthesis_prompt,
                system_prompt="You are an expert scientific writer and analyst.",
                enable_thinking=False, # Disable thinking for reliability
                timeout=120, # Give it more time
                max_tokens=max_output_tokens,
            )

            # One retry for transient failures.
            if not llm_summary or str(llm_summary).startswith("Error calling LLM API"):
                llm_summary = self.llm.generate(
                    synthesis_prompt, 
                    system_prompt="You are an expert scientific writer.",
                    enable_thinking=False,
                    timeout=30, 
                    max_tokens=1000
                )

            if not llm_summary or str(llm_summary).startswith("Error calling LLM API"):
                 user_facing_response = "⚠️ System Error: Unable to generate response from LLM."
            else:
                user_facing_response = llm_summary

        else:
            user_facing_response = "LLM features are disabled."

        # Add citations footer
        if citations:
            # Sort citations by relevance (score) just in case, though they should be sorted
            # Limit the visible citations to avoid overwhelming the user
            limit_visible = 15
            visible_citations = citations[:limit_visible]
            # Use actual newlines \n instead of escaped \\n so the frontend renders them correctly
            user_facing_response += "\n\n---\n**📚 Key Reference Sources (Top 15):**\n" + "\n".join(visible_citations)
            if len(citations) > limit_visible:
                user_facing_response += f"\n... and {len(citations) - limit_visible} more sources used in analysis."

        return user_facing_response

if __name__ == "__main__":
    agent = AgenticRAG()
    report = agent.process_query("Fix a G>A point mutation in HSC cells for in vivo delivery")
    print(report)
