import sys
import os
import time
import requests

# Add project root to Python path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.agentic_rag import AgenticRAG
except ImportError:
    # Fallback if run from root
    from core.agentic_rag import AgenticRAG

def ingest_literature(terms_list, max_per_term=5):
    print("=========================================")
    print("🚀 Bulk Literature Ingestion Script")
    print("=========================================")
    print(f"Terms to process: {len(terms_list)}")
    
    # Initialize RAG (which inits DataPipeline and loads FAISS)
    try:
        agent = AgenticRAG()
        pipeline = agent.data_pipeline
    except Exception as e:
        print(f"Error initializing RAG Agent: {e}")
        return

    total_chunks_added = 0
    
    for term in terms_list:
        print(f"\n🔍 Searching for: '{term}' (Limit: {max_per_term}) ...")
        
        # Construct a targeted query 
        # EuropePMC syntax: term AND OPEN_ACCESS:Y 
        query = f'({term}) AND OPEN_ACCESS:Y'
        
        try:
            # Step 1: Fetch Metadata
            metadata = pipeline.step1_fetch_metadata(query, max_results=max_per_term)
            
            if not metadata:
                print(f"⚠️ No results found for '{term}'")
                continue

            # Step 2: Download PDFs
            downloaded_files = pipeline.step2_download_pdfs(metadata)
            
            # Step 3: Parse Content
            parsed_docs = []
            
            # Prioritize PDFs
            if downloaded_files:
                print(f"📄 Parsing {len(downloaded_files)} PDFs...")
                parsed_docs.extend(pipeline.step3_parse_pdfs(downloaded_files))
            
            # Identify which IDs were downloaded to avoid duplicates
            downloaded_ids = set()
            for f in downloaded_files:
                # Basic filename check
                fname = os.path.basename(f)
                downloaded_ids.add(fname.split('.')[0])
                
            # Fallback to abstracts for failed downloads
            abstract_count = 0
            for m in metadata:
                doc_id = m.get('doc_id') or m.get('pmid')
                is_downloaded = False
                for d_id in downloaded_ids:
                    if doc_id and doc_id in d_id:
                        is_downloaded = True
                        break
                
                if not is_downloaded and m.get('abstract'):
                    # Create a doc structure for abstract only
                    parsed_docs.append({
                        "text": f"TITLE: {m['title']}\nJOURNAL: {m['journal']}\nYEAR: {m['year']}\nABSTRACT: {m['abstract']}", 
                        "metadata": m
                    })
                    abstract_count += 1
            
            if abstract_count > 0:
                print(f"📝 Added {abstract_count} abstracts for papers without PDFs.")
            
            if not parsed_docs:
                print(f"⚠️ No content (PDF or Abstract) extracted for '{term}'")
                continue

            # Step 3.5: Structured Extraction (Optional)
            parsed_docs = pipeline.step3_5_structured_extraction(parsed_docs)
            
            # Step 4: Chunking
            chunks = pipeline.step4_chunking(parsed_docs)
            
            # Step 5 & 6: Embed & Index
            if chunks:
                pipeline.step5_6_embed_and_index(chunks)
                total_chunks_added += len(chunks)
                print(f"✅ Indexed {len(chunks)} chunks for '{term}'")
            
        except Exception as e:
            print(f"❌ Error processing term '{term}': {e}")
        
        # Polite delay
        time.sleep(2)

    print("\n=========================================")
    print(f"🎉 Ingestion Complete.")
    print(f"Total Chunks Added to DB: {total_chunks_added}")
    print("=========================================")

if __name__ == "__main__":
    # Define a comprehensive list of gene editing topics
    topics = [
        "CRISPR-Cas9 off-target detection",
        "Prime Editing pegRNA design",
        "Base Editing cytosine adenine",
        "Cas12a Cas12b CasPhi",
        "CRISPR delivery lipid nanoparticles",
        "AAV vectors gene therapy serotypes",
        "In vivo gene editing liver muscle brain",
        "Hematopoietic stem cell gene editing sickle cell",
        "CAR-T cell engineering CRISPR",
        "Epigenetic editing CRISPRa CRISPRi",
        "Gene editing ethics safety",
        "Anti-CRISPR proteins applications",
        "High-fidelity Cas9 variants",
        "Multiplex gene editing strategies"
    ]
    
    # Run user query - increasing limit for broader coverage
    # Set max_per_term to 50 or 100 to get more papers per topic
    ingest_literature(topics, max_per_term=50)
