from core.agentic_rag import AgenticRAG
import sys

def main():
    rag = AgenticRAG()
    
    # Simple CLI loop
    print("\n🧬 Gene Editing Research Assistant (Agentic RAG) Initialized.")
    print("Type your research question below (or 'exit' to quit).")
    print("Examples:")
    print(" - 'Summarize the latest progress in Prime Editing off-target analysis'")
    print(" - 'Find papers by Zhang Feng about Cas13'")
    print(" - 'Review the efficiency of Base Editing in HSC cells'")
    
    while True:
        try:
            query = input("\n📝 User Query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            if not query.strip():
                continue
                
            response = rag.process_query(query)
            print("\n" + "="*50)
            print(response)
            print("="*50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
