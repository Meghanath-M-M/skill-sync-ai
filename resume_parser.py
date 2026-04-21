from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, DirectoryLoader

def extract_text_from_resume(pdf_path):
    """
    Extract text content from a resume PDF file.
    """
    try:
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()

        # Combine all pages into a single text string
        resume_text = ""
        for doc in docs:
            resume_text += doc.page_content + "\n"

        return resume_text.strip()

    except Exception as e:
        raise Exception(f"Error extracting text from resume: {e}")

def load_single_document(file_path):
    """Load a single PDF (using pdfplumber) or text file"""
    if file_path.endswith('.pdf'):
        # FIXED: Using the variable 'file_path' and using PDFPlumberLoader
        loader = PDFPlumberLoader(file_path) 
    else:
        loader = TextLoader(file_path)
    return loader.load()

def load_documents_from_directory(directory_path, file_type="*.pdf"):
    """Load all documents from a directory for your Knowledge Base"""
    loader = DirectoryLoader(
        directory_path,
        glob=file_type,
        # Upgraded to use PDFPlumber for bulk loading too
        loader_cls=PDFPlumberLoader if file_type == "*.pdf" else TextLoader 
    )
    return loader.load()
