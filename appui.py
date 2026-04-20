import streamlit as st
import os
import traceback

# Import the functions we built in the previous steps!
from resume_parser import extract_text_from_resume
from rag_analyser import run_skill_analyzer

# --- Page Config ---
st.set_page_config(page_title="Skill Sync AI", page_icon="🎯", layout="centered")

# --- ADD YOUR LOGO HERE ---
# Make sure the filename exactly matches what you saved it as!
# The width=200 keeps it from being massive, you can adjust this number.
# --- Side-by-Side Logo and Title ---
header_col1, header_col2 = st.columns([1, 4]) # 1 part logo, 4 parts title

with header_col1:
    st.image("logo.png",width=1200)

with header_col2:
    st.title("Skill Sync AI")
    st.markdown("Career Skill Gap Analyzer")
    st.markdown("Upload your resume and enter a target role to see how well your skills match up, and get personalized recommendations to bridge any gaps!")
st.divider()

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    st.subheader("2. Target Role")
    target_role = st.text_input("e.g., Machine Learning Intern", placeholder="Enter job title here...")

st.divider()

# --- The Magic Button ---
# When the user clicks this, the code inside runs
if st.button("🚀 Analyze My Profile", use_container_width=True):
    
    # Check if they actually uploaded a file and typed a role
    if uploaded_file is not None and target_role:
        
        # Shows a loading spinner while Llama thinks
        with st.spinner(f"Asking Llama 3 to analyze your fit for '{target_role}'..."):
            
            try:
                # 1. Streamlit holds files in memory. We need to save it temporarily 
                # so your pdfplumber script can read it from the hard drive.
                temp_pdf_path = "temp_uploaded_resume.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Extract the text using your parser
                resume_text = extract_text_from_resume(temp_pdf_path)
                
                # 3. Pass the text and role to your RAG pipeline!
                final_analysis = run_skill_analyzer(target_role, resume_text)
                
                # 4. Display the results beautifully
                st.success("Analysis Complete!")
                
                # Use st.markdown to render Llama's bullet points and bold text properly
                st.markdown("### 📊 AI Feedback")
                st.info(final_analysis)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(traceback.format_exc())
                print(traceback.format_exc())
            finally:
                # Clean up: delete the temporary PDF so it doesn't clutter your folder
                if os.path.exists("temp_uploaded_resume.pdf"):
                    os.remove("temp_uploaded_resume.pdf")
                    
    else:
        st.warning("⚠️ Please upload a resume AND enter a target role to continue.")