"""
Agentic RAG – ReAct Architecture
==================================
Upgraded from linear DAG → iterative Reason-Act loop.

Key improvements
----------------
- HyDE query expansion before retrieval
- Query decomposition for complex multi-faceted questions
- Cross-encoder reranking after FAISS retrieval (top-150 → top-25)
- RAPTOR summary tree for high-level overview / review queries
- Multi-turn conversation memory (last 3 turns injected into prompt)
- Semantic response cache (SHA-256 keyed, 7-day TTL)
- Evidence-based risk assessment (no random numbers)
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from core.knowledge_graph import GeneEditingKnowledgeGraph
from core.decision_engine import DecisionEngine
from core.risk_assessment import EvidenceBasedRiskAssessor
from core.llm_client import LLMClient
from core.data_pipeline import GeneEditingDataPipeline
from core.self_query import SelfQueryRetriever
from core.graph_reasoning import GraphReasoner
from core.review_agent import ReviewGenerator
from core.reranker import CrossEncoderReranker
from core.hyde import HypotheticalDocumentEmbedder
from core.query_decomposer import QueryDecomposer
from core.raptor import RaptorIndexer
from core.cache import QueryCache

# v2 Innovation Modules
from core.ge_almanac import GEAlmanac
from core.evidence_chain import EvidenceChain
from core.triple_graph import TripleGraph
from core.safety_firewall import SafetyFirewall
from core.u_retrieval import URetrieval
from core.semantic_chunker import SemanticChunker

# v3 Domain-Specific Innovation Modules
from core.variant_resolver import VariantResolver
from core.failure_case_db import FailureCaseDB
from core.cross_species_translator import CrossSpeciesTranslator
from core.delivery_decision_tree import DeliveryDecisionTree
from core.patent_landscape import PatentLandscape
from core.sequence_context import SequenceContext

# v4 Algorithmic Innovation Modules (paper core contributions)
from core.evidence_scorer import EvidencePyramidScorer
from core.kg_query_expander import KGQueryExpander
from core.conflict_resolver import ConflictResolver
from core.retrieval_calibrator import RetrievalCalibrator


# ---------------------------------------------------------------------------
# Synthesis prompt templates
# ---------------------------------------------------------------------------
_SYN_SYSTEM = (
    "You are an expert scientific writer and gene editing specialist. "
    "Provide structured, evidence-based, comprehensive answers. "
    "Always respond in the SAME language as the user query."
)

_SYN_USER = """Guidelines:
1. Use Markdown headers (##), bullet points, bold for key terms.
2. Synthesise ALL retrieved evidence into coherent narrative.
3. Group findings by theme (Delivery / Efficiency / Safety / Clinical).
4. Cite evidence with [n] matching the numbered sources list.
5. If conversation history is provided, maintain continuity.

{history_block}

**Retrieved Evidence & System Data:**
{dag_context}

**User Query:** {query}

Begin your comprehensive response:"""


class AgenticRAG:
    """Central orchestrator for the gene-editing knowledge assistant."""

    def __init__(self) -> None:
        self.kg = GeneEditingKnowledgeGraph()
        self.llm = LLMClient()
        self.use_llm_routing = os.getenv("USE_LLM_ROUTING", "true").lower() in (
            "1", "true", "yes"
        )

        print("Initialising Gene Editing Data Pipeline...")
        self.data_pipeline = GeneEditingDataPipeline(base_dir="data")

        self.decision_engine = DecisionEngine(knowledge_graph=self.kg)
        self.risk_assessor = EvidenceBasedRiskAssessor(kg=self.kg)

        self.self_query_retriever = SelfQueryRetriever(self.llm)
        self.graph_reasoner = GraphReasoner(self.kg.graph)
        self.review_agent = ReviewGenerator(self.llm, self.data_pipeline)

        self.reranker = CrossEncoderReranker(
            model_name=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            top_k=int(os.getenv("RERANKER_TOP_K", "20")),
        )
        self.hyde = HypotheticalDocumentEmbedder(
            self.llm,
            enabled=os.getenv("ENABLE_HYDE", "true").lower()
            in ("1", "true", "yes"),
        )
        self.decomposer = QueryDecomposer(
            self.llm,
            enabled=os.getenv("ENABLE_DECOMPOSE", "true").lower()
            in ("1", "true", "yes"),
        )
        self.cache = QueryCache(
            cache_dir="data/cache",
            ttl_seconds=int(os.getenv("CACHE_TTL", str(7 * 24 * 3600))),
        )
        self._raptor: Optional[RaptorIndexer] = None

        # ── v2 Innovation Modules ─────────────────────────────────────
        self.almanac = GEAlmanac()
        self.evidence_chain = EvidenceChain(llm_client=self.llm)
        try:
            self.triple_graph = TripleGraph()
            self.triple_graph.build_authority_from_almanac(self.almanac)
            print(f"[TripleGraph] Initialised: {self.triple_graph.stats()}")
        except Exception as e:
            print(f"[TripleGraph] Init skipped: {e}")
            self.triple_graph = None
        self.safety_firewall = SafetyFirewall(
            llm_client=self.llm, almanac=self.almanac, knowledge_graph=self.kg
        )
        self.u_retrieval = URetrieval(llm_client=self.llm)
        self.semantic_chunker = SemanticChunker()

        # ── v3 Domain-Specific Innovation Modules ─────────────────
        self.variant_resolver = VariantResolver(llm_client=self.llm, almanac=self.almanac)
        self.failure_case_db = FailureCaseDB()
        self.cross_species_translator = CrossSpeciesTranslator()
        self.delivery_decision_tree = DeliveryDecisionTree()
        self.patent_landscape = PatentLandscape()
        self.sequence_context = SequenceContext(variant_resolver=self.variant_resolver)
        print("[v3] Domain-specific modules initialised: VariantResolver, FailureCaseDB, CrossSpeciesTranslator, DeliveryDecisionTree, PatentLandscape, SequenceContext")

        # ── v4 Algorithmic Innovation Modules ─────────────────────
        self.evidence_scorer = EvidencePyramidScorer(
            alpha=float(os.getenv("EPARS_ALPHA", "0.70")),
            beta=float(os.getenv("EPARS_BETA", "0.50")),
            lam=float(os.getenv("EPARS_LAMBDA", "0.08")),
            enabled=os.getenv("ENABLE_EPARS", "true").lower() in ("1", "true", "yes"),
        )
        self.kg_expander = KGQueryExpander(
            graph=self.kg.graph,
            max_hops=int(os.getenv("KG_AQE_HOPS", "2")),
            max_expansion_terms=int(os.getenv("KG_AQE_TERMS", "8")),
            enabled=os.getenv("ENABLE_KG_AQE", "true").lower() in ("1", "true", "yes"),
        )
        self.conflict_resolver = ConflictResolver(
            cluster_threshold=float(os.getenv("CAEA_CLUSTER_TH", "0.25")),
            enabled=os.getenv("ENABLE_CAEA", "true").lower() in ("1", "true", "yes"),
        )
        self.retrieval_calibrator = RetrievalCalibrator(
            enabled=os.getenv("ENABLE_RCC", "true").lower() in ("1", "true", "yes"),
        )
        print("[v4] Algorithmic innovation modules initialised: EPARS, KG-AQE, CAEA, RCC")

    # ── Language helpers ──────────────────────────────────────────────────

    # ── Provenance RRF fusion (dict-based) ──────────────────────────────

    @staticmethod
    def _fuse_provenance(
        results_per_query: List[List[Dict]], k_rrf: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion across multiple provenance-dict lists."""
        fused: dict = {}
        for results in results_per_query:
            for rank, item in enumerate(results):
                # Use evidence string as dedup key
                uid = item.get("evidence") or item.get("text", "")[:80]
                if uid not in fused:
                    fused[uid] = dict(item)
                    fused[uid]["_rrf"] = 0.0
                fused[uid]["_rrf"] += 1.0 / (k_rrf + rank + 1)
        merged = sorted(fused.values(), key=lambda x: x["_rrf"], reverse=True)
        for m in merged:
            # Promote RRF score to the 'score' field
            m["score"] = m.pop("_rrf", m.get("score", 0.0))
        return merged

    def _detect_language(self, text: str) -> str:
        return "zh" if re.search(r"[一-鿿]", text or "") else "en"

    def _labels(self, lang: str) -> Dict:
        if lang == "zh":
            return {
                "literature": "📚 文献证据",
                "literature_empty": "本地索引中未检索到高相关证据。",
                "kg": "🔗 知识图谱",
                "decision": "⚙️ 决策引擎",
                "risk": "⚠️ 风险评估",
                "synthesis": "🤖 AI 综合结论",
            }
        return {
            "literature": "📚 Literature Evidence",
            "literature_empty": "No highly relevant local evidence found.",
            "kg": "🔗 Knowledge Graph",
            "decision": "⚙️ Decision Engine",
            "risk": "⚠️ Risk Assessment",
            "synthesis": "🤖 AI Synthesis",
        }

    # ── Intent analysis ───────────────────────────────────────────────────

    def _heuristic_intent(self, query: str) -> Dict:
        q = (query or "").lower()
        user_kw = ["upload", "import", "my paper", "上传", "导入", "我的文献"]
        return {
            "needs_retrieval": True,
            "needs_kg": any(
                k in q
                for k in [
                    "crispr", "cas9", "base editing", "prime editing",
                    "rna editing", "talen", "zfn",
                ]
            ),
            "needs_decision": any(
                k in q
                for k in [
                    "fix", "mutation", "recommend", "strategy",
                    "snv", "point mutation", "indel",
                ]
            ),
            "needs_risk": any(
                k in q
                for k in [
                    "risk", "safe", "safety", "off-target", "off target",
                    "风险", "安全",
                ]
            ),
            "is_review": any(
                k in q
                for k in [
                    "review", "overview", "survey", "综述", "总结进展",
                ]
            ),
            # Complex if: multiple ? marks, Chinese connectors, compare words, or 2+ technologies
            "is_complex": (
                q.count("?") > 1
                or q.count("？") > 1
                or any(
                    k in q
                    for k in ["compare", "versus", " vs ", "比较", "对比", "区别",
                               "difference between", "advantages and", "pros and cons"]
                )
                or len(re.findall(
                    r"\b(?:crispr|base editing|prime editing|rna editing|talen|cas9|cas13|abe|cbe)\b", q
                )) >= 2
            ),
            "search_scope": (
                "user_library"
                if any(k in q for k in user_kw)
                else "general_database"
            ),
            "target_author": None,
            "needs_almanac": any(
                k in q
                for k in [
                    "clinical trial", "approved", "fda", "ema", "治疗",
                    "疗法", "临床", "批准", "disease", "疾病", "患者",
                    "sickle", "thalassemia", "amyloidosis", "pcsk9",
                    "策略", "推荐", "选择", "哪种",
                ]
            ),
            "needs_safety_check": True,
            # v3 intent fields
            "needs_variant_resolution": VariantResolver.query_needs_resolution(q),
            "has_sequence_input": SequenceContext.query_has_sequence(q),
            "needs_delivery_advice": DeliveryDecisionTree.query_needs_delivery(q),
            "needs_failure_check": FailureCaseDB.query_needs_failure_check(q),
            "needs_patent_info": PatentLandscape.query_needs_patent(q),
            "needs_translation": CrossSpeciesTranslator.query_needs_translation(q),
        }

    def analyze_intent(self, query: str) -> Dict:
        if not self.llm.client or not self.use_llm_routing:
            return self._heuristic_intent(query)

        prompt = (
            f'''Analyse this gene-editing query and return ONLY valid JSON.

Query: "{query}"

JSON keys (all required):
- needs_retrieval: bool
- needs_kg: bool
- needs_decision: bool
- needs_risk: bool
- is_review: bool  (true if user wants a comprehensive overview/综述)
- is_complex: bool (true if query has 3+ distinct sub-questions)
- search_scope: "user_library" | "general_database"
- target_author: string | null
- needs_almanac: bool (true if query involves clinical/disease/treatment/recommendation)
- needs_safety_check: bool (always true for gene editing queries)
- needs_variant_resolution: bool (true if query asks about a specific mutation/variant editing strategy)
- has_sequence_input: bool (true if query contains a DNA/RNA sequence)
- needs_delivery_advice: bool (true if query involves delivery method selection)
- needs_failure_check: bool (true if query relates to safety risks or failures)
- needs_patent_info: bool (true if query involves patents or IP)
- needs_translation: bool (true if query involves mouse-to-human translation)'''
        )
        try:
            resp = self.llm.generate(
                prompt,
                system_prompt="JSON-only routing agent.",
                enable_thinking=False,
                timeout=15,
                max_tokens=250,
            )
            if resp and not str(resp).startswith("Error"):
                intent = self._safe_parse_json(resp)
                if intent and "needs_retrieval" in intent:
                    h = self._heuristic_intent(query)
                    if (
                        intent.get("search_scope") != "user_library"
                        and h["search_scope"] == "user_library"
                    ):
                        intent["search_scope"] = "user_library"
                    return intent
        except Exception as e:
            print(f"[Intent] LLM routing failed: {e}")
        return self._heuristic_intent(query)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _retrieve(
        self, query: str, intent: Dict, lang: str
    ) -> List[Dict]:
        is_user_scope = intent.get("search_scope") == "user_library"
        source = "user" if is_user_scope else "main"

        # 1. Translate Chinese query to English for English KB
        search_query = query
        if lang == "zh" and not is_user_scope and self.llm.client:
            try:
                t = self.llm.generate(
                    f"Translate to English for biomedical search. Output ONLY the translation: {query}",
                    system_prompt="Translator.",
                    enable_thinking=False,
                    max_tokens=120,
                    timeout=15,
                )
                if t and not t.startswith("Error"):
                    search_query = t.strip()
                    print(f"[Translate] {search_query}")
            except Exception:
                pass

        # 2. HyDE expansion
        hyde_query = self.hyde.enhanced_query(search_query)

        # 2b. KG-AQE: Knowledge-Graph-Guided Adaptive Query Expansion
        kg_expansion = self.kg_expander.expand(hyde_query)
        expanded_query = kg_expansion.get("query_expanded", hyde_query)
        kg_sub_queries = kg_expansion.get("sub_queries", [])
        if expanded_query != hyde_query:
            print(f"[KG-AQE] Expanded: +{len(kg_expansion.get('expansion_terms', []))} entities, "
                  f"depth={kg_expansion.get('expansion_depth', 0)}")

        # 3. Query decomposition for complex queries
        base_queries = (
            self.decomposer.decompose(expanded_query)
            if intent.get("is_complex")
            else [expanded_query]
        )
        # Merge KG-discovered sub-queries with decomposed queries
        sub_queries = base_queries + kg_sub_queries

        top_k = 500 if intent.get("target_author") else 150

        # 4. Retrieve for each sub-query
        #    step7_retrieve returns List[Tuple[Document, float]],
        #    we convert to provenance dicts via step8_provenance for downstream compatibility.
        all_sub_results: List[List[Dict]] = []
        for sq in sub_queries[:4]:
            try:
                raw_results = self.data_pipeline.step7_retrieve(
                    sq, top_k=top_k, source=source
                )
                # Convert (Document, score) tuples → provenance dicts
                prov_dicts = self.data_pipeline.step8_provenance(
                    raw_results, max_items=top_k
                )
                all_sub_results.append(prov_dicts)
            except Exception as e:
                print(f"[Retrieve] sub-query failed: {e}")

        if not all_sub_results:
            return []

        # Merge multiple sub-query results via provenance-dict RRF
        if len(all_sub_results) > 1:
            fused = self._fuse_provenance(all_sub_results)
        else:
            fused = all_sub_results[0]

        if not fused:
            if not is_user_scope:
                raw_fallback = self.data_pipeline.step7_retrieve(
                    search_query, top_k=50, source="user"
                )
                fused = self.data_pipeline.step8_provenance(raw_fallback, max_items=50)
            if not fused:
                return []

        # 5. Cross-encoder reranking (provenance-dict compatible)
        reranked = self.reranker.rerank_provenance(search_query, fused, top_k=25)

        # 5b. EPARS: Evidence-Pyramid-Aware Re-Scoring
        reranked = self.evidence_scorer.score(reranked, top_k=25)
        if self.evidence_scorer.enabled:
            from core.evidence_scorer import EvidencePyramidScorer
            dist = EvidencePyramidScorer.level_distribution(reranked)
            print(f"[EPARS] Evidence distribution: {dist}")

        # 6. Author filter – reranked is already List[Dict] (no step8_provenance needed)
        target_author = intent.get("target_author")
        provenance = reranked
        if target_author:
            parts = [p.lower() for p in target_author.split() if len(p) > 1]
            filtered = [
                p
                for p in provenance
                if sum(
                    1
                    for pt in parts
                    if pt
                    in (p.get("evidence", "") + p.get("text", "")).lower()
                )
                >= max(1, len(parts) - 1)
            ]
            if filtered:
                provenance = filtered

        # 7. Structured filter
        filters = self.self_query_retriever.generate_filters(query)
        if filters:
            provenance = self.self_query_retriever.apply_filters(provenance, filters)

        # 8. Acronym boosting
        acronyms = re.findall(r"\b[A-Z]{3,}\b", search_query)
        if acronyms:
            boosted = [
                p
                for p in provenance
                if any(
                    a in (p.get("text", "") + p.get("evidence", "")).upper()
                    for a in acronyms
                )
            ]
            rest = [p for p in provenance if p not in boosted]
            provenance = boosted + rest

        return provenance[:25], kg_expansion

    # ── RAPTOR helper ──────────────────────────────────────────────────────

    def _get_raptor(self) -> Optional[RaptorIndexer]:
        if self._raptor is not None:
            return self._raptor
        if self.data_pipeline.vector_store is None:
            return None
        try:
            embed_fn = lambda texts: self.data_pipeline.embeddings.embed_documents(
                texts
            )
            self._raptor = RaptorIndexer(
                llm_client=self.llm,
                embedding_fn=embed_fn,
                cache_dir="data/raptor_cache",
            )
            # Auto-build if tree cache is empty (first run after new knowledge base)
            if not self._raptor.tree:
                print("[RAPTOR] Tree is empty – auto-building from FAISS docstore...")
                try:
                    docs = list(
                        self.data_pipeline.vector_store.docstore._dict.values()
                    )
                    texts = [d.page_content for d in docs if hasattr(d, "page_content")]
                    if texts:
                        self._raptor.build(texts)
                    else:
                        print("[RAPTOR] No texts in docstore; skipping auto-build.")
                except Exception as build_err:
                    print(f"[RAPTOR] Auto-build failed: {build_err}")
        except Exception as e:
            print(f"[RAPTOR] Init failed: {e}")
        return self._raptor

    # ── Main entrypoint ────────────────────────────────────────────────────

    def process_query(
        self, query: str, history: Optional[List[Dict]] = None
    ) -> str:
        print(f"\n--- Processing Query: '{query[:80]}' ---")
        lang = self._detect_language(query)
        L = self._labels(lang)

        # Cache key includes recent history for context-awareness
        history_key = ""
        if history:
            for msg in history[-4:]:
                history_key += (
                    f"{msg.get('role','')}: {str(msg.get('content',''))[:80]}\n"
                )
        cache_key = query + "||" + history_key
        cached = self.cache.get(cache_key)
        if cached:
            return cached["response"]

        intent = self.analyze_intent(query)
        print(f"[Intent] {intent}")

        citations: List[str] = []
        dag_context = f"Query: {query}\n\n"
        provenance: List[Dict] = []
        kg_expansion: Dict = {}
        calibration: Dict = {}
        conflict_result: Dict = {}

        # ── Step 1: Retrieval ──────────────────────────────────────────────
        if intent.get("needs_retrieval", True):
            provenance, kg_expansion = self._retrieve(query, intent, lang)

            # ── Step 1-CAEA: Conflict-Aware Evidence Aggregation ──────────
            if provenance:
                conflict_result = self.conflict_resolver.resolve(provenance)
                provenance = conflict_result.get("consensus_provenance", provenance)
                n_conflicts = len(conflict_result.get("conflicts", []))
                if n_conflicts > 0:
                    print(f"[CAEA] Detected {n_conflicts} conflict(s), "
                          f"agreement={conflict_result.get('agreement_ratio', 1.0):.2f}")
                    # Inject conflict report into context for LLM awareness
                    conflict_report = ConflictResolver.format_conflict_report(
                        conflict_result, language=lang
                    )
                    dag_context += f"{conflict_report}\n\n"

            # ── Step 1-RCC: Retrieval Confidence Calibration ──────────────
            if provenance:
                kg_linked = {nid for nid, _ in kg_expansion.get("linked_entities", [])}
                calibration = self.retrieval_calibrator.calibrate(
                    query, provenance, kg_entities=kg_linked
                )
                print(f"[RCC] Confidence: {calibration.get('confidence_label', '?')} "
                      f"({calibration.get('confidence', 0):.3f}), "
                      f"signals={calibration.get('signals', {})}")

                # Inject confidence report into context
                conf_report = RetrievalCalibrator.format_confidence_report(
                    calibration, language=lang
                )
                dag_context += f"{conf_report}\n\n"

                # If low confidence, trigger additional retrieval (adaptive behaviour)
                if calibration.get("needs_deeper_retrieval") and self.llm.client:
                    print("[RCC] Low confidence → triggering deeper retrieval...")
                    try:
                        # Generate a HyDE variant with different angle
                        alt_prompt = f"Write a detailed biomedical abstract answering: {query}"
                        alt_doc = self.llm.generate(
                            alt_prompt,
                            system_prompt="Biomedical expert.",
                            enable_thinking=False,
                            timeout=15,
                            max_tokens=200,
                        )
                        if alt_doc and not str(alt_doc).startswith("Error"):
                            raw_extra = self.data_pipeline.step7_retrieve(
                                alt_doc, top_k=30, source="main"
                            )
                            extra_prov = self.data_pipeline.step8_provenance(raw_extra, max_items=30)
                            # Merge with existing provenance (dedup by evidence key)
                            existing_keys = {p.get("evidence", "") for p in provenance}
                            new_docs = [p for p in extra_prov
                                        if p.get("evidence", "") not in existing_keys]
                            if new_docs:
                                # Score new docs with EPARS
                                new_docs = self.evidence_scorer.score(new_docs, top_k=10)
                                provenance.extend(new_docs[:5])
                                print(f"[RCC] Added {min(5, len(new_docs))} supplementary documents")
                    except Exception as e:
                        print(f"[RCC] Deeper retrieval failed: {e}")

            # RAPTOR overlay for review queries
            is_review = intent.get("is_review", False)
            if is_review:
                raptor = self._get_raptor()
                if raptor and raptor.tree:
                    tree_nodes = raptor.retrieve(query, top_k=5)
                    raptor_prov = raptor.to_provenance(tree_nodes)
                    provenance = raptor_prov + provenance

            if provenance:
                dag_context += f"### {L['literature']}\n"
                for i, p in enumerate(provenance[:25], 1):
                    sd = p.get("structured_data", {})
                    sd_str = (
                        f" [Tech:{sd.get('technology','?')}"
                        f" Eff:{sd.get('efficiency','?')}]"
                        if sd
                        else ""
                    )
                    # v4: EPARS evidence level annotation
                    epars_str = ""
                    if p.get("epars_level_label"):
                        epars_str = f" [{p['epars_level_label']}]"
                    conflict_str = ""
                    if p.get("has_conflict"):
                        conflict_str = " ⚠️CONFLICTED"
                    dag_context += (
                        f"{i}) [{p['evidence']}]"
                        f" score={round(float(p.get('score', 0)), 4)}"
                        f"{epars_str}{conflict_str}{sd_str}"
                        f"\n{p['text']}\n\n"
                    )
                    citations.append(
                        f"{i}. **{p['evidence']}**"
                        f" (EPARS: {round(float(p.get('score', 0)), 2)}"
                        f", {p.get('epars_level_label', 'N/A')})"
                    )
            else:
                dag_context += f"### {L['literature']}\n{L['literature_empty']}\n\n"

            # Dedicated review mode (short-circuit to ReviewAgent + RAPTOR)
            if is_review and len(provenance) >= 3 and self.llm.client:
                print("[Mode] Review Agent")
                review_text = self.review_agent.generate_review(query, provenance)
                footer = "\n\n---\n**📚 Key Reference Sources (Top 15):**\n"
                for p in provenance[:15]:
                    footer += f"- {p['evidence']}\n"
                result = review_text + footer
                self.cache.set(cache_key, result, provenance)
                return result

        # ── Step 1b: GEAlmanac structured data ────────────────────────────
        if intent.get("needs_almanac", True):
            try:
                almanac_ctx = self.almanac.generate_almanac_context(query)
                if almanac_ctx:
                    dag_context += f"### 📋 GEAlmanac (Structured Knowledge)\n{almanac_ctx}\n\n"
                    print(f"[GEAlmanac] Injected {len(almanac_ctx)} chars of structured data")
            except Exception as e:
                print(f"[GEAlmanac] Error: {e}")

        # ── Step 1c: Triple Graph context ───────────────────────────────────
        if self.triple_graph:
            try:
                entities_for_graph = re.findall(
                    r"\b[A-Z][A-Z0-9]{1,}\b|\b(?:CRISPR|Cas\d+[a-z]?)\b", query
                )
                for ent in entities_for_graph[:2]:
                    tg_ctx = self.triple_graph.format_context_for_llm(ent, radius=2)
                    if tg_ctx:
                        dag_context += f"{tg_ctx}\n\n"
            except Exception as e:
                print(f"[TripleGraph] Query error: {e}")

        # ── Step 1d: Sequence-Aware Context ─────────────────────────────────
        seq_analysis = None
        if intent.get("has_sequence_input") or SequenceContext.query_has_sequence(query):
            try:
                seq_analysis = self.sequence_context.analyze(query)
                seq_ctx = self.sequence_context.format_context(seq_analysis)
                if seq_ctx:
                    dag_context += f"{seq_ctx}\n\n"
                    print(f"[SequenceContext] Gene: {seq_analysis.get('gene', '?')}")
            except Exception as e:
                print(f"[SequenceContext] Error: {e}")

        # ── Step 1e: Variant → Editing Strategy Resolver ──────────────────
        if intent.get("needs_variant_resolution") or VariantResolver.query_needs_resolution(query):
            try:
                resolution = self.variant_resolver.resolve(query)
                vr_ctx = self.variant_resolver.format_context(resolution)
                if vr_ctx:
                    dag_context += f"{vr_ctx}\n\n"
                    print(f"[VariantResolver] Resolved variant for {resolution.get('variant_info', {}).get('gene', '?')}")
            except Exception as e:
                print(f"[VariantResolver] Error: {e}")

        # ── Step 1f: Failure Case DB cross-check ──────────────────────────
        if intent.get("needs_failure_check") or FailureCaseDB.query_needs_failure_check(query):
            try:
                fc_ctx = self.failure_case_db.format_context(query)
                if fc_ctx:
                    dag_context += f"{fc_ctx}\n\n"
                    print("[FailureCaseDB] Injected failure case intelligence")
            except Exception as e:
                print(f"[FailureCaseDB] Error: {e}")

        # ── Step 1g: Cross-Species Translation ────────────────────────────
        if intent.get("needs_translation") or CrossSpeciesTranslator.query_needs_translation(query):
            try:
                # Extract technology and tissue from query for translation
                q_lower = query.lower()
                tech_for_trans = "CRISPR-Cas9"
                for k, v in {"base editing": "ABE", "abe": "ABE", "cbe": "CBE",
                              "prime editing": "Prime Editing", "cas9": "CRISPR-Cas9"}.items():
                    if k in q_lower:
                        tech_for_trans = v
                        break
                tissue_for_trans = "liver"
                for t in ["muscle", "brain", "retina", "lung", "hsc", "t cell"]:
                    if t in q_lower:
                        tissue_for_trans = t
                        break
                trans_ctx = self.cross_species_translator.format_context(
                    tech_for_trans, tissue_for_trans)
                if trans_ctx:
                    dag_context += f"{trans_ctx}\n\n"
                    print(f"[CrossSpecies] {tech_for_trans} / {tissue_for_trans}")
            except Exception as e:
                print(f"[CrossSpecies] Error: {e}")

        # ── Step 1h: Delivery Decision Tree ───────────────────────────────
        if intent.get("needs_delivery_advice") or DeliveryDecisionTree.query_needs_delivery(query):
            try:
                q_lower = query.lower()
                tissue_for_del = "liver"
                for t in ["muscle", "brain", "retina", "lung", "hsc", "t cell"]:
                    if t in q_lower:
                        tissue_for_del = t
                        break
                editor_for_del = None
                for e_name in ["ABE", "CBE", "PE2", "SpCas9", "SaCas9", "Cas12a"]:
                    if e_name.lower() in q_lower:
                        editor_for_del = e_name
                        break
                del_ctx = self.delivery_decision_tree.format_context(
                    tissue_for_del, editor_for_del)
                if del_ctx:
                    dag_context += f"{del_ctx}\n\n"
                    print(f"[Delivery] Tissue: {tissue_for_del}, Editor: {editor_for_del}")
            except Exception as e:
                print(f"[Delivery] Error: {e}")

        # ── Step 1i: Patent / IP Landscape ────────────────────────────────
        if intent.get("needs_patent_info") or PatentLandscape.query_needs_patent(query):
            try:
                q_lower = query.lower()
                tech_for_patent = "CRISPR-Cas9"
                for k, v in {"base editing": "ABE", "abe": "ABE", "cbe": "CBE",
                              "prime editing": "Prime Editing", "cas12": "Cas12a",
                              "rna editing": "RNA editing (ADAR)", "lnp": "LNP delivery"}.items():
                    if k in q_lower:
                        tech_for_patent = v
                        break
                pat_ctx = self.patent_landscape.format_context(tech_for_patent)
                if pat_ctx:
                    dag_context += f"{pat_ctx}\n\n"
                    print(f"[Patent] Technology: {tech_for_patent}")
            except Exception as e:
                print(f"[Patent] Error: {e}")

        # ── Step 2: Knowledge Graph ─────────────────────────────────────────
        if intent.get("needs_kg"):
            tech_map = {
                "crispr": "CRISPR KO",
                "cas9": "CRISPR KO",
                "base editing": "Base Editing",
                "prime editing": "Prime Editing",
                "rna editing": "RNA Editing ADAR",
                "cas13": "Cas13",
                "abe": "Base Editing ABE",
                "cbe": "Base Editing CBE",
            }
            q_lower = query.lower()
            found_tech = next(
                (v for k, v in tech_map.items() if k in q_lower), None
            )
            if found_tech:
                caps = self.kg.query_technology_capabilities(found_tech)
                if caps:
                    dag_context += f"### {L['kg']}\n"
                    dag_context += f"Technology: {found_tech}\n"
                    dag_context += (
                        f"Can Fix: {', '.join(caps.get('can_fix', []))}\n"
                    )
                    dag_context += (
                        f"Associated Diseases: "
                        f"{', '.join(caps.get('diseases', [])[:5])}\n"
                    )
                    dag_context += (
                        f"Target Genes: "
                        f"{', '.join(caps.get('genes', [])[:5])}\n"
                    )
                    dag_context += (
                        f"Clinical Milestones: "
                        f"{', '.join(caps.get('clinical_milestones', [])[:3])}\n"
                    )
                    dag_context += (
                        f"Indexed Studies: "
                        f"{len(caps.get('studies', []))}\n\n"
                    )

            # Entity-level KG node summary
            entities = re.findall(
                r"\b[A-Z][A-Z0-9]{1,}\b|\b(?:CRISPR|Cas\d+[a-z]?)\b", query
            )
            for ent in entities[:3]:
                summary = self.kg.get_node_summary(ent)
                if "not found" not in summary.lower():
                    dag_context += f"KG Node – {summary}\n\n"

        # ── Step 3: Decision Engine ─────────────────────────────────────────
        if intent.get("needs_decision"):
            decision_report = self.decision_engine.evaluate(query, self.llm)
            dag_context += f"### {L['decision']}\n{decision_report}\n\n"

        # ── Step 4: Risk Assessment ─────────────────────────────────────────
        if intent.get("needs_risk"):
            q_lower = query.lower()
            tech_for_risk = "CRISPR KO"
            for k, v in {
                "base editing": "Base Editing",
                "prime editing": "Prime Editing",
                "rna editing": "RNA Editing",
                "crispr": "CRISPR KO",
            }.items():
                if k in q_lower:
                    tech_for_risk = v
                    break

            delivery = "Unknown"
            for d in ["lnp", "aav", "electroporation", "mrna", "rnp"]:
                if d in q_lower:
                    delivery = d.upper()
                    break

            risk_report = self.risk_assessor.assess_risk(
                sequence="N" * 20,
                locus="query_locus",
                technology=tech_for_risk,
                delivery_system=delivery,
            )
            dag_context += (
                f"### {L['risk']} ({tech_for_risk})\n"
                f"- Risk Level: **{risk_report['risk_level']}**"
                f" (score={risk_report['overall_risk_score']}"
                f" {risk_report['uncertainty_interval']})\n"
                f"- Mechanism: {risk_report['mechanism']} |"
                f" Indel rate: {risk_report['indel_rate']}\n"
                f"- Clinical Safety: {risk_report['clinical_safety']}\n"
                f"- Evidence: {risk_report['evidence_sources'][0]}\n"
                f"- Recommendations:"
                f" {'; '.join(risk_report['recommendations'][:2])}\n\n"
            )

        # ── Step 5: Conversation history string ─────────────────────────────
        history_str = ""
        if history:
            for msg in history[-6:]:
                role = msg.get("role", "")
                content = str(msg.get("content", ""))[:400]
                history_str += f"{role.capitalize()}: {content}\n"

        # ── Step 6: LLM Synthesis ────────────────────────────────────────────
        history_block = (
            f"**Conversation History:**\n{history_str}\n" if history_str else ""
        )
        synthesis_prompt = _SYN_USER.format(
            history_block=history_block,
            dag_context=dag_context[:8000],
            query=query,
        )

        max_tokens = 4000 if len(provenance) > 10 else 2500
        user_facing_response = ""

        if self.llm.client:
            user_facing_response = self.llm.generate(
                synthesis_prompt,
                system_prompt=_SYN_SYSTEM,
                enable_thinking=False,
                timeout=120,
                max_tokens=max_tokens,
            )
            if not user_facing_response or str(user_facing_response).startswith(
                "Error"
            ):
                user_facing_response = self.llm.generate(
                    synthesis_prompt,
                    system_prompt=_SYN_SYSTEM,
                    enable_thinking=False,
                    timeout=60,
                    max_tokens=1200,
                )
            if not user_facing_response or str(user_facing_response).startswith(
                "Error"
            ):
                user_facing_response = (
                    "⚠️ LLM synthesis unavailable. "
                    "Structured evidence retrieved – see sources below."
                )
        else:
            user_facing_response = (
                f"**{L['synthesis']}**: LLM disabled.\n\n"
                f"**Retrieved {len(provenance)} relevant chunks** from knowledge base.\n"
                "Enable LLM (set OPENAI_API_KEY) for synthesised answers."
            )

        # ── Citations footer ─────────────────────────────────────────────────
        if citations:
            limit = 15
            user_facing_response += (
                "\n\n---\n**📚 Key Reference Sources (Top 15):**\n"
            )
            user_facing_response += "\n".join(citations[:limit])
            if len(citations) > limit:
                user_facing_response += (
                    f"\n… and {len(citations) - limit} more sources used."
                )

        # ── Step 7: Evidence Chain — inline citation refinement ───────────
        if provenance and self.llm.client:
            try:
                cited_answer, ref_block = self.evidence_chain.insert_citations(
                    query, user_facing_response, provenance, language=lang
                )
                if cited_answer and not str(cited_answer).startswith("Error"):
                    user_facing_response = cited_answer
                    print("[EvidenceChain] Citations inserted")
            except Exception as e:
                print(f"[EvidenceChain] Citation insertion skipped: {e}")

        # ── Step 8: Safety Firewall — post-generation check ──────────────
        safety_result = None
        if intent.get("needs_safety_check", True):
            try:
                safety_result = self.safety_firewall.run_safety_check(
                    response=user_facing_response,
                    provenance=provenance,
                    faithfulness_score=-1,
                )
                verdict = safety_result.get("safety_verdict", "PASSED")
                print(f"[SafetyFirewall] Verdict: {verdict}")

                # Inject disclaimer if needed
                user_facing_response = SafetyFirewall.inject_disclaimer(
                    user_facing_response, safety_result, language=lang
                )

                # Append safety report for flagged/blocked responses
                if verdict in ("FLAGGED", "BLOCKED"):
                    safety_report = self.safety_firewall.format_safety_report(safety_result)
                    user_facing_response += f"\n\n---\n{safety_report}"
            except Exception as e:
                print(f"[SafetyFirewall] Check failed: {e}")

        # ── Step 9: RCC Uncertainty Injection ─────────────────────────────
        if calibration:
            user_facing_response = RetrievalCalibrator.inject_uncertainty(
                user_facing_response, calibration, language=lang
            )

        # ── Cache store ──────────────────────────────────────────────────────
        self.cache.set(cache_key, user_facing_response, provenance)

        return user_facing_response

    # ── Utilities ─────────────────────────────────────────────────────────

    def _safe_parse_json(self, text: str):
        if not text:
            return None
        cleaned = re.sub(r"```[a-z]*", "", str(text)).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None


if __name__ == "__main__":
    agent = AgenticRAG()
    report = agent.process_query(
        "What is the off-target risk of base editing for SCD treatment?"
    )
    print(report)
