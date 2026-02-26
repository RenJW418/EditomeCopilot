import os
from pypdf import PdfReader

class DocumentLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_documents(self):
        """
        Loads all .txt and .pdf files from the data directory.
        Returns a list of text chunks/documents.
        """
        documents = []
        if not os.path.exists(self.data_dir):
            return documents

        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            
            if filename.endswith(".txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        documents.append(f.read())
                    print(f"Loaded TXT: {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(filepath)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        documents.append(text)
                    print(f"Loaded PDF: {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
        return documents
