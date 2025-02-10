import streamlit as st
import requests
import time
import os
import numpy as np
import json
import base64
from sentence_transformers import SentenceTransformer

# ---------------------------
# Azure Computer Vision (OCR) Config
# ---------------------------
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", st.secrets.get("AZURE_ENDPOINT"))
AZURE_SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY", st.secrets.get("AZURE_SUBSCRIPTION_KEY"))

# ---------------------------
# Azure OpenAI Config (using AzureOpenAI from OpenAI package)
# ---------------------------
from openai import AzureOpenAI

# Get Azure OpenAI config from environment/Streamlit secrets
endpoint_url = os.getenv("ENDPOINT_URL", st.secrets.get("ENDPOINT_URL"))
deployment = os.getenv("DEPLOYMENT_NAME", st.secrets.get("DEPLOYMENT_NAME"))
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", st.secrets.get("AZURE_OPENAI_API_KEY"))

# Initialize the Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint_url,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# ---------------------------
# Helper Functions
# ---------------------------
def extract_text_from_image(image_bytes):
    """
    Use Azure's Computer Vision Read API to extract text from an image.
    """
    analyze_url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_SUBSCRIPTION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    
    response = requests.post(analyze_url, headers=headers, data=image_bytes)
    response.raise_for_status()
    
    # The API returns an Operation-Location header which contains the URL to poll for the result.
    operation_url = response.headers["Operation-Location"]
    
    # Poll for the result until it's "succeeded"
    while True:
        result_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_SUBSCRIPTION_KEY})
        result_response.raise_for_status()
        result = result_response.json()
        if result.get("status") == "succeeded":
            break
        time.sleep(1)
    
    # Parse and extract text from the result (across all pages)
    extracted_text = ""
    read_results = result.get("analyzeResult", {}).get("readResults", [])
    for page in read_results:
        for line in page.get("lines", []):
            extracted_text += line.get("text", "") + " "
    return extracted_text.strip()

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def llm_judgement(question, total_weightage, student_answer, expected_answer, similarity):
    """
    Uses Azure OpenAI (via the AzureOpenAI client) to provide a final evaluation,
    including a score and detailed feedback.
    """
    prompt_text = f"""
You are an expert educator evaluating student exam answers. Please review the following details and provide a final evaluation.

Question: {question}
Total Marks: {total_weightage}

Expected Answer:
{expected_answer}

Student Answer:
{student_answer}

The cosine similarity between the student's answer and the expected answer is {similarity:.2f} (0 indicates no similarity, and 1 indicates perfect similarity).

Based on this information, please:
1. Assign a final score between 0 and {total_weightage}.
2. Provide detailed feedback highlighting what was done well, what key points may be missing, and any suggestions for improvement.

Your response should be in the following format:
Final Score: <score>
Feedback: <detailed feedback>
    """
    
    # Prepare messages in the format expected by Azure OpenAI.
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert educator evaluating student exam answers."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ]
        }
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
            stop=None,
            stream=False
        )
        # The client returns a response object with a to_json() method.
        response_json = json.loads(completion.to_json())
        judgement = response_json["choices"][0]["message"]["content"]
    except Exception as e:
        judgement = f"Error retrieving LLM judgement: {e}"
    return judgement

def grade_answer_with_llm(question, total_weightage, student_answer, expected_answer):
    """
    Hybrid grading:
    - Compute a baseline score using cosine similarity (with sentence transformer embeddings).
    - Use Azure OpenAI to provide a detailed evaluation and final feedback.
    
    Returns:
    - baseline_score: Numeric score from cosine similarity.
    - final_judgement: LLM's final evaluation.
    """
    # Load the sentence transformer model (this may take a moment)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for expected and student answers
    expected_embedding = model.encode(expected_answer)
    student_embedding = model.encode(student_answer)
    
    # Compute cosine similarity
    similarity = cosine_similarity(student_embedding, expected_embedding)
    
    # Baseline score: scale the similarity by total_weightage (capped to total_weightage)
    baseline_score = min(similarity * total_weightage, total_weightage)
    
    # Use Azure OpenAI to provide detailed feedback and a final score
    final_judgement = llm_judgement(question, total_weightage, student_answer, expected_answer, similarity)
    
    return baseline_score, final_judgement

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("Hybrid Grading POC (Azure OpenAI)")
st.write("Upload an image of a handwritten answer. The app will extract text using Azure OCR and grade it using a hybrid approach.")

# Input fields for question details
question = st.text_area("Enter the Question:", "Explain the process of photosynthesis.")
expected_answer = st.text_area("Enter the Expected Answer:", 
"""Photosynthesis is the process by which plants convert light energy into chemical energy.
It involves the absorption of light by chlorophyll, the conversion of carbon dioxide and water
into glucose, and the release of oxygen as a byproduct.""")
total_weightage = st.number_input("Total Marks for the Question:", min_value=1, max_value=100, value=10)

# File uploader for the student's handwritten answer image
uploaded_file = st.file_uploader("Upload an Image of the Handwritten Answer:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Process and Grade"):
        with st.spinner("Extracting text from image using Azure OCR..."):
            image_bytes = uploaded_file.read()
            try:
                extracted_text = extract_text_from_image(image_bytes)
            except Exception as e:
                st.error(f"Error during OCR extraction: {e}")
                extracted_text = ""
        
        if extracted_text:
            st.subheader("Extracted Student Answer:")
            st.write(extracted_text)
            
            # Grade the answer using our hybrid grading function
            with st.spinner("Grading the answer..."):
                baseline_score, final_judgement = grade_answer_with_llm(
                    question, total_weightage, extracted_text, expected_answer
                )
            
            st.subheader("Results:")
            st.write(f"**Baseline Score (from cosine similarity):** {baseline_score:.2f} out of {total_weightage}")
            st.write("**Azure OpenAI Final Evaluation:**")
            st.write(final_judgement)
        else:
            st.error("No text could be extracted from the image.")
