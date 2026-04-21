from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

def run_skill_analyzer(target_role, resume_text, db_path="faiss_index"):
    print(f"Starting analysis for role: {target_role}")
    
    # 1. Load the Embedding Model and FAISS Database
    print("Loading vector database...")
    try:
        # FIXED: Changed 'model' to 'model_name' for HuggingFace syntax
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        raise

    try:
        # FIXED: Relies entirely on the relative path "faiss_index"
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS database loaded successfully")
    except Exception as e:
        print(f"❌ Error loading FAISS database: {e}")
        raise

    # 2. Initialize Groq LLM
    print("Connecting to Groq API...")
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192", 
        )
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        raise 

    # 3. Design the Prompt Template
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
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "input"]
    ).partial(resume_context=resume_text)

    # 4. Create the Retrieval Chain
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
        # FIXED: Extracts the plain text content from the Groq response object
        return response.content 
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise
