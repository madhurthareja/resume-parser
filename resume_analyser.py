import streamlit as st
from groq import Groq
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Union
import json
import os
from pypdf import PdfReader
import pandas as pd

# Define Pydantic models
class ResumeData(BaseModel):
    name: str
    email: str
    phone: Optional[str]
    skills: List[str]
    education: Union[str, List[dict]]
    experience: Union[str, List[dict]]

class JobData(BaseModel):
    title: str
    requiredSkills: List[str]
    requiredEducation: Optional[str]
    requiredExperience: Optional[str]

# Define nested model for skills_match
class SkillsMatch(BaseModel):
    matched: List[str]
    missing: List[str]
    percentage: float

# Define model for structured analysis response
class AnalysisData(BaseModel):
    skills_match: SkillsMatch
    education_fit: str
    experience_fit: str
    suitability_score: int
    summary: str

# Define model for bulk resume extraction
class BulkResumeData(BaseModel):
    name: str
    email: str
    college: Optional[str]
    highest_education: Optional[str]
    research_interests: Optional[Union[str, List[str], List[dict]]]
    prior_work: Optional[Union[str, List[str], List[dict]]]
    publications_or_projects: Optional[Union[str, List[str], List[dict]]]

# Initialize Streamlit app
st.title("Resume-Job Match Analyzer")
st.markdown("Upload a resume PDF and enter a job description to get a realistic, evidence-based analysis of fit.")

# Set up Groq client
api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY. Add it to .streamlit/secrets.toml or export it as an environment variable.")
    st.stop()
client = Groq(api_key=api_key)  # Use Streamlit secrets or environment variable

MODEL_NAME = "llama-3.1-8b-instant"

# # Sidebar for API key input (optional, for local testing)
# with st.sidebar:
#     api_key = st.text_input("Enter Gemini API Key (optional if set in secrets)", type="password")
#     if api_key:
#         client = genai.Client(api_key=api_key)

# Tabs
tab_match, tab_bulk = st.tabs(["Match Analyzer", "Bulk Extractor"])

# Session state for storing JSONs, messages, and analysis
if "resume_json" not in st.session_state:
    st.session_state.resume_json = None
if "job_json" not in st.session_state:
    st.session_state.job_json = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# Function to process resume PDF
def process_resume(file):
    try:
        # Extract text locally from PDF for Groq
        reader = PdfReader(file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        resume_text = "\n".join(pages_text).strip()

        if not resume_text:
            st.error("Could not extract text from the PDF. Try a different file or a text-based PDF.")
            return None, None

        prompt = (
            "Extract the details from the resume text. "
            "Include the following fields: name, email, phone, skills (as a list), education, and experience. "
            "Return only valid JSON with those fields. Use empty strings for missing fields. "
            "If education or experience are lists, return them as arrays of objects.\n\n"
            f"Resume Text:\n{resume_text}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You extract structured data and output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw_json = response.choices[0].message.content
        resume_data = ResumeData.model_validate_json(raw_json)

        # Normalize fields to strings for downstream prompts/UI
        if isinstance(resume_data.education, list):
            resume_data.education = json.dumps(resume_data.education, ensure_ascii=True)
        if isinstance(resume_data.experience, list):
            resume_data.experience = json.dumps(resume_data.experience, ensure_ascii=True)
        resume_data.phone = resume_data.phone or ""

        return resume_data, raw_json

    except (ValidationError, json.JSONDecodeError) as e:
        st.error(f"Failed to parse resume JSON: {e}")
        return None, None

def _list_or_str_to_text(value: Optional[Union[str, List[str], List[dict]]]) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            if isinstance(item, dict):
                normalized_items.append(json.dumps(item, ensure_ascii=True))
            else:
                normalized_items.append(str(item))
        return ", ".join([item for item in normalized_items if item.strip()])
    return str(value)

def process_resume_bulk(file):
    try:
        file.seek(0)
        reader = PdfReader(file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        resume_text = "\n".join(pages_text).strip()

        if not resume_text:
            return None, "Could not extract text from the PDF."

        prompt = (
            "Extract the details from the resume text. "
            "Include the following fields: name, email, college, highest_education, research_interests, prior_work, publications_or_projects. "
            "Return only valid JSON with those fields. Use empty strings for missing fields. "
            "Use lists for research_interests, prior_work, and publications_or_projects when needed. "
            "For prior_work or publications_or_projects, you may return lists of objects with keys like title, organization, role, and duration.\n\n"
            f"Resume Text:\n{resume_text}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You extract structured data and output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw_json = response.choices[0].message.content
        resume_data = BulkResumeData.model_validate_json(raw_json)

        return {
            "Name": resume_data.name,
            "Email": resume_data.email,
            "College": resume_data.college or "",
            "Highest Education": resume_data.highest_education or "",
            "Research Interests": _list_or_str_to_text(resume_data.research_interests),
            "Experience/Prior Work": _list_or_str_to_text(resume_data.prior_work),
            "Publications or Key Projects": _list_or_str_to_text(resume_data.publications_or_projects),
        }, None

    except (ValidationError, json.JSONDecodeError) as e:
        return None, f"Failed to parse resume JSON: {e}"

# Function to process job description
def process_job_description(text):
    try:
        # Use Gemini to structure job description
        prompt = (
            f"Convert the following job description into structured JSON with fields: title, requiredSkills (as a list), requiredEducation, and requiredExperience. "
            f"Use empty strings for missing fields.\n\n"
            f"Job Description: {text}"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You extract structured data and output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw_json = response.choices[0].message.content
        job_data = JobData.model_validate_json(raw_json)
        job_data.requiredEducation = job_data.requiredEducation or ""
        job_data.requiredExperience = job_data.requiredExperience or ""
        return job_data, raw_json
    except Exception as e:
        st.error(f"Error processing job description: {e}")
        return None, None

# Function to analyze match holistically using Gemini
def analyze_match(resume, job):
    if not resume or not job:
        return None
    
    try:
        # Convert resume and job data to strings for the prompt
        resume_str = json.dumps(resume.model_dump(), indent=2)
        job_str = json.dumps(job.model_dump(), indent=2)
        
        # Define a balanced prompt for holistic analysis
        prompt = (
            "You are a fair and evidence-based hiring reviewer. Evaluate the candidate’s resume against the job description. "
            "Be direct and realistic, but not needlessly harsh. Always note strengths as well as gaps. "
            "Consider:\n"
            "- Skills: Which ones match, which are missing, and how critical the gaps are.\n"
            "- Education: Does it meet the job’s needs or is there a mismatch?\n"
            "- Experience: Is it sufficient in depth, relevance, and years?\n"
            "- Overall fit: Can this candidate do the role with reasonable onboarding?\n\n"
            f"Resume Data:\n{resume_str}\n\n"
            f"Job Description Data:\n{job_str}\n\n"
            "Provide a structured JSON response with:\n"
            "- skills_match: Object with 'matched' (list of matched skills), 'missing' (list of missing skills), 'percentage' (number, e.g., 66.7).\n"
            "- education_fit: String, balanced assessment.\n"
            "- experience_fit: String, balanced assessment.\n"
            "- suitability_score: Integer (0-100), calibrated to the role level. Avoid defaulting low scores.\n"
            "- summary: String, include at least one strength and one gap."
        )

        # Make API call for analysis
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You analyze and output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw_json = response.choices[0].message.content
        return AnalysisData.model_validate_json(raw_json)

    except Exception as e:
        st.error(f"Error analyzing match: {e}")
        return None

# Function to answer user questions using Gemini
def answer_question(question, resume, job, analysis):
    if not resume or not job or not question:
        return "Please process the resume and job description first, and enter a question."
    
    try:
        # Convert resume, job data, and analysis to strings for the prompt
        resume_str = json.dumps(resume.model_dump(), indent=2)
        job_str = json.dumps(job.model_dump(), indent=2)
        analysis_str = json.dumps(analysis.model_dump(), indent=2) if analysis else "No analysis available."
        
        # Define a prompt for answering the question with balanced tone
        prompt = (
            "You are a fair, evidence-based hiring reviewer. Answer the question about the candidate’s suitability for the job based on their resume, job description, and previous analysis. "
            "Be direct and clear, but balanced. If the question is broad, give a calibrated judgment with a key strength and a key gap. "
            "If specific, cite evidence and avoid extremes unless the data clearly supports it.\n\n"
            f"Resume Data:\n{resume_str}\n\n"
            f"Job Description Data:\n{job_str}\n\n"
            f"Previous Analysis:\n{analysis_str}\n\n"
            f"Question: {question}\n\n"
            "Return the response as a plain text string."
        )

        # Make API call for the answer
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You answer clearly and directly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error answering question: {e}")
        return "Failed to answer the question due to an error."

with tab_match:
    # File uploader for resume PDF
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

    # Text area for job description
    job_description = st.text_area("Enter Job Description", placeholder="e.g., Senior Software Engineer requiring Python, React, AWS, Bachelor's in CS, 3+ years experience")

    # Process button
    if st.button("Analyze Match"):
        if resume_file and job_description:
            with st.spinner("Processing resume..."):
                resume_data, resume_raw = process_resume(resume_file)
                if resume_data:
                    st.session_state.resume_json = resume_data.model_dump()
            
            with st.spinner("Processing job description..."):
                job_data, job_raw = process_job_description(job_description)
                if job_data:
                    st.session_state.job_json = job_data.model_dump()
            
            if st.session_state.resume_json and st.session_state.job_json:
                # JSONs are processed but not displayed
                pass
                
                # Analyze match
                with st.spinner("Analyzing match..."):
                    analysis = analyze_match(
                        ResumeData(**st.session_state.resume_json),
                        JobData(**st.session_state.job_json)
                    )
                    st.session_state.last_analysis = analysis

                # Display dashboard
                if analysis:
                    st.subheader("Match Analysis Dashboard")
                    
                    # Layout with columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Skills Match Table
                        st.markdown("**Skills Match**")
                        skills_data = {
                            "Matched Skills": ", ".join(analysis.skills_match.matched) or "None",
                            "Missing Skills": ", ".join(analysis.skills_match.missing) or "None",
                            "Match Percentage": f"{analysis.skills_match.percentage:.1f}%"
                        }
                        st.dataframe(skills_data, use_container_width=True)
                        
                        # Education and Experience Fit
                        st.markdown("**Education Fit**")
                        st.write(analysis.education_fit)
                        st.markdown("**Experience Fit**")
                        st.write(analysis.experience_fit)
                    
                    with col2:
                        # Suitability Score
                        st.markdown("**Suitability Score**")
                        st.metric(label="Score (0-100)", value=analysis.suitability_score)
                    
                    # Summary
                    st.markdown("**Summary**")
                    st.write(analysis.summary)
                    
                    # Add to messages for chat history
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Analysis completed: Suitability Score {analysis.suitability_score}/100"
                    })
        else:
            st.error("Please upload a resume PDF and enter a job description.")

    # Chat-like interface
    st.subheader("Match Analysis Q&A")
    question = st.text_input("Ask a question about the candidate (e.g., 'Is this candidate suitable for the job?' or 'Does the candidate have Python skills?')")
    if st.button("Submit Question"):
        if question and st.session_state.resume_json and st.session_state.job_json:
            resume = ResumeData(**st.session_state.resume_json)
            job = JobData(**st.session_state.job_json)
            analysis = st.session_state.last_analysis
            
            # Get answer from Groq
            with st.spinner("Generating answer..."):
                answer = answer_question(question, resume, job, analysis)
            
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "system", "content": answer})
        else:
            st.error("Please process the resume and job description first, and enter a question.")

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with tab_bulk:
    st.subheader("Bulk Resume Extractor")
    st.markdown("Upload multiple PDFs to extract key fields and download a CSV.")

    bulk_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Extract to CSV"):
        if not bulk_files:
            st.error("Please upload at least one PDF.")
        else:
            rows = []
            errors = []
            progress = st.progress(0)
            total = len(bulk_files)

            for idx, file in enumerate(bulk_files, start=1):
                data, err = process_resume_bulk(file)
                if data:
                    rows.append(data)
                else:
                    errors.append(f"{file.name}: {err}")
                progress.progress(idx / total)

            if errors:
                st.warning("Some files could not be processed:")
                for message in errors:
                    st.write(f"- {message}")

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="resume_extracts.csv",
                    mime="text/csv",
                )