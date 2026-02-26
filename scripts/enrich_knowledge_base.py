
import json
import os
import sys
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.literature_fetcher import PubMedFetcher

# Initialize 
fetcher = PubMedFetcher()
DB_PATH = "data/knowledge_base/literature_db_GEA_v2026_Q1.json"

def enrich_db():
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    print(f"Loading database from {DB_PATH}...")
    with open(DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records: {len(data)}")
    
    # Identify records needing update (missing 'authors' or empty 'authors')
    to_update = []
    for i, record in enumerate(data):
        if "authors" not in record or not record["authors"] or record["authors"] == "Unknown":
            # Only if it has a valid-looking numeric ID
            if record.get("id", "").isdigit():
                to_update.append(i)
    
    print(f"Records needing author info: {len(to_update)}")
    
    # Process in batches
    # Reduced batch size to 20 to avoid timeouts
    BATCH_SIZE = 20
    updated_count = 0
    
    # For demo purposes, limit total to 500. 
    # To run FULL database, change this to: LIMIT = len(to_update)
    LIMIT = 500 
    
    batch_indices = []
    
    for i, idx in enumerate(to_update[:LIMIT]):
        batch_indices.append(idx)
        
        if len(batch_indices) >= BATCH_SIZE:
            print(f"Processing batch {(i+1)//BATCH_SIZE} (IDs {i-BATCH_SIZE+1} to {i})...")
            process_batch(data, batch_indices)
            updated_count += len(batch_indices)
            batch_indices = []
            time.sleep(1.0) # Respect rate limits
            
    # Process remaining
    if batch_indices:
        process_batch(data, batch_indices)
        updated_count += len(batch_indices)

    print(f"Updated {updated_count} records.")
    
    # Save back
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Database saved.")

def process_batch(data, indices):
    ids = [data[i]["id"] for i in indices]
    print(f"Fetching metadata for batch of {len(ids)} IDs...")
    
    try:
        detailed_articles = fetcher.fetch_article_details(ids)
        
        # Create a map for quick lookup
        details_map = {art["pmid"]: art for art in detailed_articles}
        
        for idx in indices:
            record = data[idx]
            pmid = record["id"]
            if pmid in details_map:
                details = details_map[pmid]
                # Update fields
                if "authors" in details and details["authors"]:
                    record["authors"] = details["authors"] # List of strings
                if "journal" in details and details["journal"]:
                    record["journal"] = details["journal"]
                if "year" in details and details["year"] != "Unknown Year":
                     record["year"] = details["year"]
                     
    except Exception as e:
        print(f"Batch failed: {e}")

if __name__ == "__main__":
    enrich_db()
