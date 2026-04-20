from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

def run_skill_analyzer(target_role, resume_text, db_path="faiss_index"):
    print(f"Starting analysis for role: {target_role}")
    
    # Test LLM first
    print("Testing LLM connectivity...")
    if not test_llm():
        raise Exception("LLM test failed - Ollama may not be responding")
    
    # Use hardcoded absolute path to ensure FAISS index is found
    db_path = r"c:\Projects\faiss_index"
    
    # 1. Load the Embedding Model and FAISS Database
    print("Loading vector database...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        print("✅ Embeddings model loaded")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        raise

    # Note: allow_dangerous_deserialization is required by FAISS for local loading
    try:
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS database loaded successfully")
    except Exception as e:
        print(f"❌ Error loading FAISS database: {e}")
        raise

    # 2. Initialize Llama (Assuming you are using Ollama)
    print("Waking up Llama...")
    try:
        # Change "llama3:latest" to match the exact model name you pulled in Ollama
        llm = Ollama(model="llama3:latest", temperature=0.1, timeout=120)  # Increased timeout to 2 minutes
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        raise 

    # 3. Design the Prompt Template
    # This is where we force Llama to format the output exactly how you want it
    template = """
    You are an expert technical recruiter and career coach. 
    Use the following pieces of retrieved job descriptions to analyze the candidate's resume against their target role.
    
    Retrieved Job Descriptions & Requirements:
    {context}
    
    Candidate's Target Role: {input}
    
    Candidate's Resume Text:
    {resume_context}
    
    Based strictly on the retrieved job descriptions above, provide the following structured analysis:
    1. Match Score (Percentage)
    2. Matching Skills (Bullet points)
    3. Missing Skills (Bullet points)
    4. Learning Recommendations (Short, actionable steps)
    5. Explanation (A brief sentence explaining why a specific missing skill is important based on the job descriptions)
    
    Analysis:
    """
    
    # We pass 'resume_context' as a partial variable so it injects dynamically
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "input"]
    ).partial(resume_context=resume_text)

    # 4. Create the Retrieval Chain
    # This automatically handles taking the question, searching FAISS, and passing it to the prompt
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    # 5. Run the Analysis
    print(f"Analyzing resume against '{target_role}' requirements...\n")
    try:
        response = qa_chain.invoke(target_role)
        print("✅ Analysis completed successfully")
        return response
    except AssertionError as e:
        print(f"❌ FAISS dimension mismatch: {e}")
        raise Exception(
            "FAISS index dimension does not match current embeddings.\n"
            "Please rebuild the FAISS database with the current embedding model."
        )
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_llm():
    """Test if LLM is working"""
    try:
        llm = Ollama(model="llama3:latest", temperature=0.1, timeout=30)
        response = llm.invoke("Say 'Hello' in one word.")
        print(f"LLM test response: {response}")
        return True
    except Exception as e:
        print(f"LLM test failed: {e}")
        return False
    
    # Make sure Ollama is running in your terminal before executing this!
    
    print("============= AI ANALYSIS =============")
    print(final_output)