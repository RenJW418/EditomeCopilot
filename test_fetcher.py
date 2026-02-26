
from core.literature_fetcher import PubMedFetcher
import json

def test_fetch():
    fetcher = PubMedFetcher()
    ids = ["32393822"] 
    print(f"Fetching details for: {ids}")
    try:
        results = fetcher.fetch_article_details(ids)
        print(f"Got {len(results)} results.")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Fetcher raised: {e}")

if __name__ == "__main__":
    test_fetch()
