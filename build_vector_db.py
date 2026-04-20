from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def create_and_save_vector_db(documents, save_path="faiss_index"):
    """
    Takes loaded documents, splits them, embeds them, and saves to FAISS.
    """
    # 1. Initialize the Text Splitter
    # chunk_size is characters. 500 is roughly a good sized paragraph.
    # chunk_overlap ensures a sentence isn't cut cleanly in half losing context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50 
    )
    
    # Split the documents
    print(f"Splitting {len(documents)} documents...")
    
    # Clean metadata to avoid deepcopy issues with PDFPlumberLoader
    cleaned_docs = []
    for doc in documents:
        clean_metadata = {
            "source": doc.metadata.get("source", "unknown")
        }
        cleaned_docs.append(Document(page_content=doc.page_content, metadata=clean_metadata))
    
    chunks = text_splitter.split_documents(cleaned_docs)
    print(f"Successfully created {len(chunks)} text chunks.")

    # 2. Initialize the Embedding Model
    # Using Ollama with nomic-embed-text model
    print("Loading embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # 3. Create the FAISS Vector Database
    print("Converting text to vectors and building FAISS database...")
    print("This may take several minutes on first run...")
    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
        print("✅ FAISS database created successfully!")
    except Exception as e:
        print(f"❌ Error creating FAISS database: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 4. Save the database locally
    # We save it so we don't have to re-process the PDFs every time the app runs
    print(f"Saving FAISS database to '{save_path}'...")
    try:
        vector_db.save_local(save_path)
        print(f"✅ FAISS database successfully saved to the '{save_path}' directory!")
    except Exception as e:
        print(f"❌ Error saving FAISS database: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return vector_db

# --- Example of how to tie it together with your previous code ---
if __name__ == "__main__":
    from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
    
    # Assuming you have a folder called 'job_descriptions' with PDFs
    # Replace this with your actual loading logic from earlier
    print("Loading documents...")
    try:
        # Use the raw string prefix 'r' before the path in Windows to avoid slash issues
        loader = DirectoryLoader(r'C:\Projects\job_descriptions', glob="*.pdf", loader_cls=PDFPlumberLoader)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} documents")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    if len(docs) > 0:
        # Build and save the database
        try:
            create_and_save_vector_db(docs)
            print("✅ Vector database creation completed successfully!")
        except Exception as e:
            print(f"❌ Error in vector database creation: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print("No documents found to process. Check your directory path!")