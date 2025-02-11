import streamlit as st
import requests
import time
import os
import numpy as np
import json
import asyncio
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# ---------------------------
# Azure Computer Vision (OCR) Config
# ---------------------------
AZURE_ENDPOINT = 'https://visiontestq.cognitiveservices.azure.com/'
AZURE_SUBSCRIPTION_KEY = '8uBAdXzutVprzehOfze86B8NQFZFFLJLmZxPcEeZROlCU2LPQ9cxJQQJ99BBACYeBjFXJ3w3AAAFACOGSteD'

# ---------------------------
# Azure OpenAI Config
# ---------------------------
endpoint_url = 'https://cog-dfecc4yseotvi.openai.azure.com/'
deployment = 'gpt-4o'
subscription_key = '549b98ee2a85468d8611749e24eac5fe'

client = AzureOpenAI(
    azure_endpoint=endpoint_url,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# ---------------------------
# Optimization: Cache Sentence Transformer Model
# ---------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_transformer()

# ---------------------------
# Helper Functions
# ---------------------------
def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def async_extract_text_from_image(image_bytes):
    """Use Azure OCR asynchronously to extract text from an image."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_text_from_image, image_bytes)

def extract_text_from_image(image_bytes):
    """Azure OCR API call to extract text."""
    analyze_url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_SUBSCRIPTION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    
    response = requests.post(analyze_url, headers=headers, data=image_bytes)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    
    while True:
        result_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_SUBSCRIPTION_KEY})
        result_response.raise_for_status()
        result = result_response.json()
        if result.get("status") == "succeeded":
            break
        time.sleep(1)
    
    extracted_text = " ".join(line.get("text", "") for page in result.get("analyzeResult", {}).get("readResults", []) for line in page.get("lines", []))
    return extracted_text.strip()

async def async_grade_answer(question, total_weightage, student_answer, expected_answer):
    """Asynchronous grading function to prevent blocking UI."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, grade_answer_with_llm, question, total_weightage, student_answer, expected_answer)

def grade_answer_with_llm(question, total_weightage, student_answer, expected_answer):
    """Hybrid grading using cosine similarity & Azure OpenAI."""
    expected_embedding = model.encode(expected_answer).astype(np.float32)
    student_embedding = model.encode(student_answer).astype(np.float32)
    
    similarity = cosine_similarity(student_embedding, expected_embedding)
    baseline_score = min(similarity * total_weightage, total_weightage)
    
    final_judgement = llm_judgement(question, total_weightage, student_answer, expected_answer, similarity)
    
    return baseline_score, final_judgement

def llm_judgement(question, total_weightage, student_answer, expected_answer, similarity):
    """Azure OpenAI LLM grading."""
    prompt_text = f"""
You are an expert educator evaluating student exam answers. Please review the details and provide a final evaluation.

Question: {question}
Total Marks: {total_weightage}

Expected Answer:
{expected_answer}

Student Answer:
{student_answer}

Cosine similarity between expected and student answer: {similarity:.2f} (0 = no similarity, 1 = perfect).

Please:
1. Assign a final score (0 to {total_weightage}).
2. Provide detailed feedback on correctness, missing points, and improvement.

Format:
Final Score: <score>
Feedback: <detailed feedback>
    """
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert educator evaluating student exam answers."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=250,
            temperature=0.5,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False
        )
        response_json = json.loads(completion.to_json())
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error retrieving LLM judgement: {e}"

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("📄 Hybrid Grading System (OCR + AI)")
st.write("Upload a handwritten answer image to get AI-based grading.")

# Store extracted text and results to avoid unnecessary recomputation
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "grading_result" not in st.session_state:
    st.session_state.grading_result = None

# User Inputs
question = st.text_area("Enter the Question:", "Explain photosynthesis.")
expected_answer = st.text_area("Expected Answer:", "Photosynthesis is the process where plants convert light energy...")
total_weightage = st.number_input("Total Marks:", min_value=1, max_value=100, value=10)
uploaded_file = st.file_uploader("Upload Answer Image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("Extract Text"):
        with st.spinner("🔍 Extracting text from image..."):
            image_bytes = uploaded_file.read()
            extracted_text = extract_text_from_image(image_bytes)
            if extracted_text:
                st.session_state.ocr_text = extracted_text
            else:
                st.error("❌ No text detected. Try again!")

if st.session_state.ocr_text:
    st.subheader("📝 Extracted Text:")
    st.write(st.session_state.ocr_text)
    
    if st.button("Grade Answer"):
        with st.spinner("📊 Evaluating answer..."):
            baseline_score, final_judgement = grade_answer_with_llm(
                question, total_weightage, st.session_state.ocr_text, expected_answer
            )
            st.session_state.grading_result = (baseline_score, final_judgement)

if st.session_state.grading_result:
    baseline_score, final_judgement = st.session_state.grading_result
    st.subheader("🎯 Results:")
    st.write(f"**Baseline Score:** {baseline_score:.2f} / {total_weightage}")
    st.write("**AI Final Evaluation:**")
    st.write(final_judgement)
