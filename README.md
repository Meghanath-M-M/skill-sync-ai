# Skill Sync AI 🚀

A Retrieval-Augmented Generation (RAG) application that analyzes a candidate's resume against a target job description to find skill gaps and provide learning recommendations. 

### 🛠️ Tech Stack
* **LLM:** Meta Llama 3.1 (via Groq API)
* **Vector Database:** FAISS 
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Framework:** LangChain
* **Frontend/Deployment:** Streamlit Community Cloud

### 🌟 Features
* Parses PDF resumes and extracts text.
* Uses FAISS to perform similarity searches against a database of technical job descriptions.
* Leverages Llama 3.1 to generate Match Scores, Missing Skills, and Actionable Learning Paths.

🌐 **[Try the Live App Here](https://skill-sync-ai-rbtkcgscdulvpfstlslqer.streamlit.app/)**
