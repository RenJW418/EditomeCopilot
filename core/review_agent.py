from core.llm_client import LLMClient
import re

class ReviewGenerator:
    """
    Autonomous Agent for generating comprehensive scientific reviews.
    """
    def __init__(self, llm_client: LLMClient, retrieval_pipeline):
        self.llm = llm_client
        self.retriever = retrieval_pipeline

    def generate_review(self, topic: str, provenances: list) -> str:
        """
        Synthesizes a structured review paper section based on retrieved literatures.
        """
        if not provenances:
            return "Insufficient literature found to generate a review."

        print(f"\n[Review Agent] Generating review for topic: '{topic}' with {len(provenances)} papers.")
        
        # 1. Clustering / Grouping (Simplified by LLM for now)
        # We feed the abstracts and ask LLM to organize them first.
        
        papers_text = ""
        for i, p in enumerate(provenances):
            papers_text += f"[{i+1}] {p['evidence']}\nSummary: {p['text'][:400]}...\n\n"

        prompt = f"""
        You are a senior editor at Nature Reviews Genetics.
        Write a comprehensive logical review on the topic: "{topic}".
        
        IMPORTANT: The user queried in a specific language. You MUST respond in that SAME language.
        If the query "{topic}" is in Chinese, the entire review must be in Chinese.
        
        Use the following provided literature evidence to support your writing.
        Integrate the citations naturally (e.g., [1], [2]).
        
        Structure Requirements:
        1. **Title**: Catchy and academic (In the target language).
        2. **Executive Summary**: 3-sentence high-level overview.
        3. **Key Advances**: Group the literature into logical themes (e.g., "Efficiency Improvements", "Off-target Analysis", "Clinical Translations").
        4. **Comparative Analysis**: Compare different approaches found in the text.
        5. **Future Directions**: synthesis of where the field is going based on these papers.
        
        Literature Evidence:
        {papers_text}
        
        Output formatted in Markdown.
        """
        
        try:
            review = self.llm.generate(
                prompt, 
                system_prompt="You are an expert scientific writer.",
                max_tokens=2000,
                enable_thinking=False # Disable thinking for reliability if it times out
            )
            return review
        except Exception as e:
            return f"Error generating review: {e}"
