import streamlit as st
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# =========================================================
# SKILL DATABASE
# =========================================================
SKILLS_DB = [
    "python","java","c++","sql","mysql","postgresql","mongodb",
    "machine learning","deep learning","nlp","pandas","numpy",
    "tensorflow","pytorch","flask","django","fastapi",
    "aws","azure","gcp","docker","kubernetes","ci/cd",
    "git","linux","rest api","data analysis","data science"
]

# =========================================================
# LOAD MODEL (runs only once)
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# =========================================================
# FUNCTIONS
# =========================================================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def clean_text(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\.\-\+ ]', '', text)
    return text.strip()

def extract_skills(text):
    return {skill for skill in SKILLS_DB if skill in text}

def extract_experience(text):
    matches = re.findall(r'(\d{1,2}(?:\.\d)?)\s*(?:years|year|yrs|yr)', text)
    years = [float(x) for x in matches if float(x) <= 40]
    return max(years) if years else 0

def extract_required_experience(text):
    range_match = re.findall(r'(\d{1,2})\s*-\s*(\d{1,2})\s*years', text)
    if range_match:
        return float(range_match[0][0])

    matches = re.findall(r'(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|year|yrs|yr)', text)
    years = [float(x) for x in matches if float(x) <= 40]
    return max(years) if years else 0

def calculate_final_score(semantic, skill, exp_match):
    exp_score = {"High":100, "Medium":70, "Low":30, "Not Specified":50}.get(exp_match,50)
    return (semantic*0.4) + (skill*0.4) + (exp_score*0.2)

def hiring_decision(score):
    if score >= 80: return "Strong Shortlist"
    elif score >= 65: return "Shortlist"
    elif score >= 50: return "Consider"
    else: return "Reject"

def generate_feedback(missing_skills, exp_match, semantic_score):
    suggestions = []

    if missing_skills:
        suggestions.append("Add these skills: " + ", ".join(list(missing_skills)[:6]))

    if exp_match == "Low":
        suggestions.append("Add internships/projects to prove real experience")

    if semantic_score < 55:
        suggestions.append("Rewrite summary aligned to job description keywords")

    suggestions.append("Add measurable achievements (numbers & impact)")
    return suggestions

def highlight_resume(text, matched_skills, missing_skills):
    highlighted = text

    for skill in matched_skills:
        pattern = re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
        highlighted = pattern.sub(
            f"<span style='background-color:#b6fcb6;padding:2px;border-radius:3px'>{skill}</span>",
            highlighted
        )

    missing_html = ""
    if missing_skills:
        missing_html = "<b>Missing Keywords:</b><br>"
        for skill in missing_skills:
            missing_html += f"<span style='background-color:#ffb3b3;padding:3px;margin:2px;border-radius:3px'>{skill}</span> "

    return highlighted, missing_html

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "analysis_done": False,
    "score": None,
    "matched": None,
    "missing": None,
    "extra": None,
    "skill_score": None,
    "candidate_exp": None,
    "required_exp": None,
    "exp_match": None,
    "final_score": None,
    "decision": None,
    "feedback": None,
    "raw_resume": "",
    "highlighted_resume": "",
    "missing_panel": ""
}

for key,val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================
# UI
# =========================================================
st.title("Level 5 ATS Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")
analyze = st.button("Analyze Resume")

# =========================================================
# ANALYSIS
# =========================================================
if analyze:

    if uploaded_file is None or job_description.strip() == "":
        st.warning("Please upload resume and paste job description")

    else:
        with st.spinner("Running ATS Analysis..."):

            raw_resume = extract_text_from_pdf(uploaded_file)
            resume_text = clean_text(raw_resume)
            jd_text = clean_text(job_description)

            st.session_state.raw_resume = raw_resume

            # Semantic similarity
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            jd_emb = model.encode(jd_text, convert_to_tensor=True)
            similarity = util.cos_sim(resume_emb, jd_emb)
            st.session_state.score = float(similarity[0][0]) * 100

            # Skill analysis
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            st.session_state.matched = resume_skills & jd_skills
            st.session_state.missing = jd_skills - resume_skills
            st.session_state.extra = resume_skills - jd_skills

            st.session_state.skill_score = (
                len(st.session_state.matched)/len(jd_skills)*100 if jd_skills else 0
            )

            # Experience
            cand_exp = extract_experience(resume_text)
            req_exp = extract_required_experience(jd_text)

            if req_exp == 0: exp_match="Not Specified"
            elif cand_exp >= req_exp: exp_match="High"
            elif cand_exp >= req_exp*0.6: exp_match="Medium"
            else: exp_match="Low"

            st.session_state.candidate_exp = cand_exp
            st.session_state.required_exp = req_exp
            st.session_state.exp_match = exp_match

            # Final decision
            st.session_state.final_score = calculate_final_score(
                st.session_state.score,
                st.session_state.skill_score,
                st.session_state.exp_match
            )

            st.session_state.decision = hiring_decision(st.session_state.final_score)

            # Feedback
            st.session_state.feedback = generate_feedback(
                st.session_state.missing,
                st.session_state.exp_match,
                st.session_state.score
            )

            # Highlighted resume
            highlighted, missing_html = highlight_resume(
                st.session_state.raw_resume,
                st.session_state.matched,
                st.session_state.missing
            )

            st.session_state.highlighted_resume = highlighted
            st.session_state.missing_panel = missing_html

            st.session_state.analysis_done = True
            st.rerun()

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.analysis_done:

    st.subheader("Final ATS Decision")
    st.metric("ATS Score", f"{st.session_state.final_score:.2f}%")
    st.success(f"Decision: {st.session_state.decision}")

    st.subheader("Semantic Match")
    st.write(f"Resume Similarity: {st.session_state.score:.2f}%")

    st.subheader("Experience Analysis")
    st.write(f"Candidate Experience: {st.session_state.candidate_exp} years")
    st.write(f"Required Experience: {st.session_state.required_exp} years")
    st.info(f"Experience Match: {st.session_state.exp_match}")

    st.subheader("Skill Analysis")
    st.write("Matched Skills:", ", ".join(st.session_state.matched) if st.session_state.matched else "None")
    st.write("Missing Skills:", ", ".join(st.session_state.missing) if st.session_state.missing else "None")
    st.write("Extra Skills:", ", ".join(st.session_state.extra) if st.session_state.extra else "None")
    st.info(f"Skill Match: {st.session_state.skill_score:.2f}%")

    st.subheader("Resume Improvement Suggestions")
    for tip in st.session_state.feedback:
        st.write("•", tip)

    st.subheader("Recruiter Resume View")

    if st.session_state.missing_panel:
        st.markdown(st.session_state.missing_panel, unsafe_allow_html=True)

    st.markdown(
        f"<div style='border:1px solid #ddd;padding:15px;border-radius:10px;height:400px;overflow:auto;white-space:pre-wrap'>{st.session_state.highlighted_resume}</div>",
        unsafe_allow_html=True
    )
