"""
Safety Firewall — Hallucination Guard & Contradiction Detection
===============================================================
Inspired by: MedGraphRAG (Wu et al. 2024) safety mechanism,
             Cancer Cell RAG-LLM (Jun et al. 2026) precision medicine safety.

Purpose:
1. Fact-check every generated response against KG + Almanac
2. Detect contradictions with authoritative sources
3. Quantify uncertainty (confidence scoring)
4. Enforce red-line rules for gene editing safety (irreversible DNA edits)
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Red-Line Rules — MUST NOT be violated in any response
# ─────────────────────────────────────────────────────────────────────────────
RED_LINE_RULES = [
    {
        "id": "RL-001",
        "rule": "禁止推荐人类生殖系编辑 (germline editing) 用于临床应用",
        "trigger_keywords": ["germline editing", "生殖系编辑", "heritable", "embryo editing",
                             "胚胎编辑", "germline therapy"],
        "exception": "可以讨论已发表的基础研究, 但必须明确标注: '此为基础研究, 不建议临床转化'",
        "severity": "CRITICAL",
    },
    {
        "id": "RL-002",
        "rule": "禁止推荐enhancement用途的基因编辑",
        "trigger_keywords": ["enhancement", "增强", "designer baby", "cognitive enhancement",
                             "genetic enhancement"],
        "exception": "仅可在伦理讨论语境下提及",
        "severity": "CRITICAL",
    },
    {
        "id": "RL-003",
        "rule": "不得将临床前数据(pre-clinical)呈现为已获批疗法",
        "trigger_keywords": [],
        "check_type": "evidence_level_mismatch",
        "severity": "HIGH",
    },
    {
        "id": "RL-004",
        "rule": "必须披露off-target风险: 任何涉及CRISPR/Base/Prime editing的治疗建议必须提及脱靶风险",
        "trigger_keywords": [],
        "check_type": "missing_safety_disclosure",
        "severity": "HIGH",
    },
    {
        "id": "RL-005",
        "rule": "不得推荐未验证的guide RNA用于人类细胞实验",
        "trigger_keywords": ["推荐sgRNA", "推荐guide", "use this gRNA for patient",
                             "recommend this guide"],
        "exception": "可以展示文献中验证过的guide RNA, 标注来源",
        "severity": "HIGH",
    },
    {
        "id": "RL-006",
        "rule": "不得声称100%编辑效率或零脱靶",
        "trigger_keywords": ["100% efficiency", "100%效率", "zero off-target",
                             "no off-target", "零脱靶", "完全无脱靶"],
        "severity": "HIGH",
    },
]


class SafetyFirewall:
    """Post-generation safety checker for gene-editing RAG system."""

    def __init__(self, llm_client=None, almanac=None, knowledge_graph=None):
        """
        Parameters
        ----------
        llm_client : LLMClient, optional
            For LLM-based contradiction detection.
        almanac : GEAlmanac, optional
            For fact-checking against structured data.
        knowledge_graph : KnowledgeGraph or TripleGraph, optional
            For KG-based consistency checking.
        """
        self.llm = llm_client
        self.almanac = almanac
        self.kg = knowledge_graph

    # ──────────────────────────────────────────────────────────────────────
    # 1. Red-Line Check (fast, rule-based)
    # ──────────────────────────────────────────────────────────────────────
    def check_red_lines(self, response: str) -> List[Dict[str, Any]]:
        """
        Check response against red-line rules.

        Returns list of violations. Empty list = all clear.
        """
        violations = []
        response_lower = response.lower()

        for rule in RED_LINE_RULES:
            # Keyword-based rules
            triggered_keywords = [
                kw for kw in rule.get("trigger_keywords", [])
                if kw.lower() in response_lower
            ]
            if triggered_keywords:
                violations.append({
                    "rule_id": rule["id"],
                    "rule": rule["rule"],
                    "severity": rule["severity"],
                    "triggered_by": triggered_keywords,
                    "exception": rule.get("exception", ""),
                })

        return violations

    # ──────────────────────────────────────────────────────────────────────
    # 2. Contradiction Detection (LLM-based)
    # ──────────────────────────────────────────────────────────────────────
    def detect_contradictions(
        self,
        response: str,
        provenance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use LLM to detect contradictions between the response and
        retrieved evidence.

        Returns
        -------
        dict with contradictions list and consistency_score.
        """
        if not self.llm or not hasattr(self.llm, "generate"):
            return {"contradictions": [], "consistency_score": -1,
                    "error": "No LLM available"}

        evidence_text = "\n".join(
            f"[{i+1}] {p.get('title', 'N/A')}: "
            f"{p.get('evidence', p.get('text', p.get('abstract', '')))[:300]}"
            for i, p in enumerate(provenance[:8])
        )

        prompt = (
            "你是一个基因编辑领域的事实核查员。\n"
            "请对比以下 [回答] 和 [证据], 找出所有矛盾之处。\n\n"
            "矛盾类型:\n"
            "1. 数据矛盾 — 回答中的数字(效率、脱靶率等)与证据不符\n"
            "2. 关系矛盾 — 回答中的因果/适用关系与证据相反\n"
            "3. 时间矛盾 — 回答声称的时间线与证据不符\n"
            "4. 过度声明 — 回答中的确定性语气超出证据支持范围\n\n"
            f"## 回答\n{response[:2000]}\n\n"
            f"## 证据\n{evidence_text}\n\n"
            '请输出JSON数组, 每个矛盾为一个对象:\n'
            '[{"type": "数据矛盾|关系矛盾|时间矛盾|过度声明", '
            '"claim_in_answer": "...", '
            '"evidence_says": "...", '
            '"severity": "HIGH|MEDIUM|LOW"}]\n'
            '如果没有矛盾, 输出空数组 []'
        )

        result = self.llm.generate(prompt, timeout=45, max_tokens=2048)
        if result.startswith("Error"):
            return {"contradictions": [], "consistency_score": -1, "error": result}

        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            contradictions = json.loads(json_match.group()) if json_match else []
        except (json.JSONDecodeError, AttributeError):
            contradictions = []

        # Score
        severity_weights = {"HIGH": 0.3, "MEDIUM": 0.15, "LOW": 0.05}
        penalty = sum(
            severity_weights.get(c.get("severity", "LOW"), 0.05)
            for c in contradictions
        )
        consistency_score = max(0.0, 1.0 - penalty)

        return {
            "contradictions": contradictions,
            "consistency_score": round(consistency_score, 3),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 3. Almanac Fact-Check (structured verification)
    # ──────────────────────────────────────────────────────────────────────
    def check_against_almanac(self, response: str) -> List[Dict[str, Any]]:
        """
        Check specific claims against GEAlmanac structured data.
        Fast, deterministic, no LLM needed.
        """
        warnings = []
        if not self.almanac:
            return warnings

        resp_lower = response.lower()

        # Check: Does response claim something is FDA-approved that isn't?
        approved_drugs = {a["drug"].lower() for a in self.almanac.approvals}
        approval_patterns = re.findall(
            r'(fda[已\s]*(approved|批准|获批)|获得.{0,10}批准)',
            resp_lower,
        )
        if approval_patterns:
            # Verify the drug/technology mentioned is actually approved
            for trial in self.almanac.clinical_trials:
                if trial["name"].lower() in resp_lower:
                    if "approved" not in trial.get("status", "").lower():
                        warnings.append({
                            "type": "false_approval_claim",
                            "entity": trial["name"],
                            "actual_status": trial["status"],
                            "severity": "HIGH",
                        })

        # Check: Efficiency claims within known bounds
        efficiency_claims = re.findall(r'(\d{1,3})%\s*(editing|efficiency|效率|编辑)', resp_lower)
        for pct_str, _ in efficiency_claims:
            pct = int(pct_str)
            if pct > 99:
                warnings.append({
                    "type": "unrealistic_efficiency",
                    "claim": f"{pct}% efficiency",
                    "context": "No gene editing technology achieves 100% efficiency in vivo",
                    "severity": "HIGH",
                })

        return warnings

    # ──────────────────────────────────────────────────────────────────────
    # 4. Uncertainty Quantification
    # ──────────────────────────────────────────────────────────────────────
    def quantify_uncertainty(
        self,
        response: str,
        provenance: List[Dict[str, Any]],
        faithfulness_score: float = -1,
    ) -> Dict[str, Any]:
        """
        Estimate confidence based on multiple signals.

        Returns dict with confidence_level, confidence_score, flags.
        """
        flags = []
        score = 1.0

        # Signal 1: Provenance quality
        if not provenance:
            score -= 0.4
            flags.append("No provenance sources found")
        else:
            avg_score = sum(p.get("score", 0) for p in provenance) / len(provenance)
            if avg_score < 0.3:
                score -= 0.2
                flags.append(f"Low average retrieval score: {avg_score:.3f}")

            # Source diversity
            years = [p.get("year", p.get("pub_year", 0)) for p in provenance if p.get("year") or p.get("pub_year")]
            if years:
                latest = max(int(y) for y in years if str(y).isdigit())
                if latest < 2022:
                    score -= 0.1
                    flags.append(f"Most recent source is from {latest} — may be outdated")

        # Signal 2: Faithfulness (from evidence chain)
        if faithfulness_score >= 0:
            if faithfulness_score < 0.5:
                score -= 0.3
                flags.append(f"Low faithfulness score: {faithfulness_score}")
            elif faithfulness_score < 0.8:
                score -= 0.1
                flags.append(f"Moderate faithfulness: {faithfulness_score}")

        # Signal 3: Hedging language detection
        hedge_patterns = [
            r'可能|或许|也许|不确定|有待验证|尚未证实|初步研究|需要更多研究',
            r'may|might|potentially|uncertain|preliminary|further research needed',
        ]
        hedge_count = sum(len(re.findall(p, response, re.I)) for p in hedge_patterns)
        if hedge_count > 5:
            score -= 0.05
            flags.append(f"High hedging language ({hedge_count} instances)")

        # Signal 4: Response length vs. evidence
        if len(response) > 3000 and len(provenance) < 3:
            score -= 0.1
            flags.append("Long response with few sources — possible over-generation")

        score = max(0.0, min(1.0, score))

        # Classify
        if score >= 0.8:
            level = "HIGH"
        elif score >= 0.5:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "confidence_level": level,
            "confidence_score": round(score, 3),
            "flags": flags,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 5. Full Safety Pipeline
    # ──────────────────────────────────────────────────────────────────────
    def run_safety_check(
        self,
        response: str,
        provenance: List[Dict[str, Any]],
        faithfulness_score: float = -1,
    ) -> Dict[str, Any]:
        """
        Run the full safety pipeline:
        1. Red-line violations
        2. Almanac fact-check
        3. LLM contradiction detection
        4. Uncertainty quantification
        5. Final verdict

        Returns
        -------
        dict with all sub-results and a final safety_verdict.
        """
        # 1. Red lines (fast)
        red_line_violations = self.check_red_lines(response)

        # 2. Almanac check (fast)
        almanac_warnings = self.check_against_almanac(response)

        # 3. Contradiction detection (LLM, slower)
        contradiction_result = self.detect_contradictions(response, provenance)

        # 4. Uncertainty
        uncertainty = self.quantify_uncertainty(
            response, provenance, faithfulness_score
        )

        # 5. Final verdict
        critical_violations = [v for v in red_line_violations if v["severity"] == "CRITICAL"]
        high_issues = (
            [v for v in red_line_violations if v["severity"] == "HIGH"] +
            [w for w in almanac_warnings if w.get("severity") == "HIGH"] +
            [c for c in contradiction_result.get("contradictions", []) if c.get("severity") == "HIGH"]
        )

        if critical_violations:
            verdict = "BLOCKED"
            verdict_reason = f"Critical red-line violation(s): {', '.join(v['rule_id'] for v in critical_violations)}"
        elif len(high_issues) >= 2:
            verdict = "FLAGGED"
            verdict_reason = f"{len(high_issues)} high-severity issues detected"
        elif uncertainty["confidence_level"] == "LOW":
            verdict = "LOW_CONFIDENCE"
            verdict_reason = "Insufficient evidence support"
        else:
            verdict = "PASSED"
            verdict_reason = "All checks passed"

        return {
            "safety_verdict": verdict,
            "verdict_reason": verdict_reason,
            "red_line_violations": red_line_violations,
            "almanac_warnings": almanac_warnings,
            "contradictions": contradiction_result,
            "uncertainty": uncertainty,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 6. Format Safety Report for Display
    # ──────────────────────────────────────────────────────────────────────
    def format_safety_report(self, safety_result: Dict[str, Any]) -> str:
        """Format safety check result as Markdown for the user."""
        verdict = safety_result.get("safety_verdict", "UNKNOWN")
        verdict_emoji = {
            "PASSED": "✅", "LOW_CONFIDENCE": "⚠️",
            "FLAGGED": "🚨", "BLOCKED": "🛑",
        }.get(verdict, "❓")

        lines = [f"### {verdict_emoji} Safety Check: **{verdict}**"]
        lines.append(f"*{safety_result.get('verdict_reason', '')}*\n")

        # Red-line violations
        violations = safety_result.get("red_line_violations", [])
        if violations:
            lines.append("**Red-Line Violations:**")
            for v in violations:
                lines.append(f"- 🛑 [{v['rule_id']}] {v['rule']}")
                if v.get("exception"):
                    lines.append(f"  *Exception: {v['exception']}*")

        # Almanac warnings
        warnings = safety_result.get("almanac_warnings", [])
        if warnings:
            lines.append("\n**Fact-Check Warnings:**")
            for w in warnings:
                lines.append(f"- ⚠️ {w['type']}: {w.get('entity', '')} — actual: {w.get('actual_status', w.get('context', ''))}")

        # Contradictions
        contradictions = safety_result.get("contradictions", {}).get("contradictions", [])
        if contradictions:
            lines.append("\n**Contradictions Detected:**")
            for c in contradictions:
                lines.append(f"- [{c.get('severity', '?')}] {c.get('type', 'Unknown')}")
                lines.append(f"  Answer claims: {c.get('claim_in_answer', '?')}")
                lines.append(f"  Evidence says: {c.get('evidence_says', '?')}")

        # Confidence
        unc = safety_result.get("uncertainty", {})
        conf = unc.get("confidence_score", -1)
        if conf >= 0:
            lines.append(f"\n**Confidence Score:** {conf:.1%} ({unc.get('confidence_level', '')})")
            for flag in unc.get("flags", []):
                lines.append(f"- {flag}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # 7. Inject Safety Disclaimer
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def inject_disclaimer(
        response: str, safety_result: Dict[str, Any], language: str = "zh"
    ) -> str:
        """
        If safety check flagged issues, prepend a disclaimer banner
        to the response.
        """
        verdict = safety_result.get("safety_verdict", "PASSED")
        if verdict == "PASSED":
            return response

        if verdict == "BLOCKED":
            if language == "zh":
                disclaimer = (
                    "⚠️ **安全警告**: 此回答触发了关键安全规则，部分内容可能不适合直接使用。"
                    "请咨询专业人员。\n\n"
                )
            else:
                disclaimer = (
                    "⚠️ **Safety Warning**: This response triggered critical safety rules "
                    "and may not be suitable for direct use. Please consult experts.\n\n"
                )
        elif verdict == "FLAGGED":
            if language == "zh":
                disclaimer = (
                    "⚠️ **注意**: 本回答中部分声明与已知证据存在不一致，已标注。"
                    "请交叉验证后使用。\n\n"
                )
            else:
                disclaimer = (
                    "⚠️ **Note**: Some claims in this response may be inconsistent with "
                    "known evidence. Please cross-check before use.\n\n"
                )
        else:  # LOW_CONFIDENCE
            if language == "zh":
                disclaimer = "ℹ️ *本回答的证据支持度较低，请谨慎参考。*\n\n"
            else:
                disclaimer = "ℹ️ *This response has limited evidence support. Use with caution.*\n\n"

        return disclaimer + response
