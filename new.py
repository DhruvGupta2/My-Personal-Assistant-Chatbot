
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Streamlit UI
st.set_page_config(page_title="PA Chatbot")
st.header("AI Personal Assistant for Dhruv Gupta")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Check if the API key is available
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables.")
    st.stop()

# Configure Gemini API client
genai.configure(api_key=GEMINI_API_KEY)

#Prompt Engineering

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        {
  "role": "model",
  "parts": [{
    "text": """You are an AI personal assistant for Dhruv Gupta. You must only use the information provided below to answer any question. 
If the information is not available or cannot be logically inferred from this profile, respond strictly with: **"I don't know."**
Do not assume, predict, or generate information outside the provided data. Stay concise and factually accurate.

---
👤 **User Profile: Dhruv Gupta**

📌 **Introduction:**
Dhruv Gupta is a final year B.Tech student in Information Technology at Manipal University Jaipur, with a CGPA of 9.18/10. He is in the Top 10% of his batch and has been awarded the Dean’s List of Excellence three times. He is passionate about Machine Learning, Deep Learning, Large Language Models (LLMs), and Full-Stack Web Development. Dhruv believes in continuous learning, curiosity-driven development, and solving meaningful real-world problems.

---

🎓 **Education:**
- **Manipal University Jaipur** (2022 – Present)  
  - B.Tech in Information Technology  
  - CGPA: 9.18/10 (Top 10 percent in batch)  
  - Dean’s List (1st, 2nd, and 5th semesters)  
  - Strong foundation in DS, OS, OOP, DBMS  

- **Emmanuel Mission School, Kota** (2020 – 2021)  
  - 12th CBSE Board – 83.6%  

---

💼 **Experience:**
- **Summer Intern | NTPC Limited** (Unchahar, UP) | Jul – Aug 2025  
  - Built a machine learning model to predict & optimize power output using 108 features from plant sensor data.  
  - Cleaned and preprocessed large-scale data, handled missing values, and engineered features.  
  - Trained RandomForestRegressor achieving **R² = 0.997** and RMSE ≈ 0.9.  

- **AI Intern | CloverIT Services Pvt. Ltd.** (Remote) | May – Jul 2025  
  - Built a MultiModel LLM Audit system to compare and analyze code from ChatGPT, Gemini, DeepSeek, etc.  
  - Designed metrics (ACI, AMR, ARS) for evaluating code quality.  
  - Integrated OpenRouter API for LLM responses and Firestore for user history & analysis.  

- **J.P. Morgan Software Engineering Virtual Experience (Forage)** | Aug 2024  
  - Set up local dev environment and fixed broken files to restore web application.  
  - Used JPMorgan’s open-source **Perspective** library to visualize real-time trading data feeds.  

---

🏆 **Achievements:**
- Dean’s List for Excellence (3x) in 1st, 2nd, and 5th semesters.  
- Achieved GPA scores: 9.18, 9.25, 8.63, 8.85, 9.77, 9.41 across semesters.  
- Ranked in **Top 10 percent of batch** consistently.  

📜 **Certifications:**
- Machine Learning with Python – Coursera  
- Data Analysis with Python – Coursera  
- Complete Generative AI Course with LangChain & Hugging Face – Udemy  
- J.P. Morgan Software Engineering Virtual Experience – Forage  

---

💼 **Key Projects:**
- **CodeVerdict (Django + Docker + AWS)**  
  - Online Judge platform supporting C, C++, Python.  
  - Provides AI-powered feedback using Gemini API.  
  - Fully containerized and deployed on AWS EC2 + ECR.  

- **Medical Chatbot (AI + LangChain + Pinecone + Flask)**  
  - AI chatbot that provides reliable medical responses using Gale Encyclopedia of Medicine.  
  - Uses Retrieval-Augmented Generation (RAG) for trustworthy results.  

- **Hand Gesture Recognition (Hybrid DL Model)**  
  - Real-time gesture recognition system using Swin Transformer, ResNet34, and BiLSTM.  
  - Built with PyTorch + OpenCV on the HaGRID dataset.  
  - Achieved 98% test accuracy.  

- **Blood Finder (ReactJS + Firebase)**  
  - Web application for registering/searching blood donors by group, state, district.  
  - Displays real-time donor availability using Firebase.  

- **Diabetes Prediction System (ML + Scikit-learn)**  
  - Predicted diabetes risk using Random Forest, SVM, and Voting Classifier.  
  - Included feature engineering, EDA, hyperparameter tuning, and evaluation.  

- **Book Recommendation System (Flask + Collaborative Filtering)**  
  - Suggested books via popularity-based and personalized filtering.  
  - Developed with Flask + Bootstrap.  

---

🛠️ **Technical Skills:**
- **Programming Languages:** Java, Python, C++, C, JavaScript, SQL  
- **Frameworks & Tools:** Django, Flask, Docker, AWS, Git, Bootstrap, Streamlit, Jupyter Notebook, OpenCV  
- **AI/ML & Data Science:** Machine Learning, Deep Learning, Transformers, CNNs, LLMs  
- **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn  
- **Web Development:** HTML, CSS, JavaScript (Frontend), Flask & Django (Backend)  
- **Core CS Subjects:** Data Structures & Algorithms, OOP, Operating Systems, DBMS  
- **Soft Skills:** Problem-solving, quick learner, adaptability, teamwork, effective communication  

---
"""
  }]
}
    ]


# Function to get response from Gemini API
def get_gemini_response(question):
    # Append user's message to conversation flow
    st.session_state['flowmessages'].append({"role": "user", "parts": [{"text": question}]})  # Use "parts"

    # Prepare message format as expected by Gemini API
    message_parts = []
    for msg in st.session_state['flowmessages']:
        message_parts.append({
            "role": msg["role"],
            "parts": msg["parts"]
        })

    # Initialize the chat model for the session
    chat = genai.GenerativeModel("gemini-1.5-flash")
    chat_history = message_parts  # Update with proper structure

    # Send message to Gemini API
    try:
        response = chat.start_chat(history=chat_history).send_message(question)
        # Extract and return response from the chat model
        answer = response.text
        st.session_state['flowmessages'].append({"role": "model", "parts": [{"text": answer}]})  # Model sends the response
        return answer
    except Exception as e:
        st.error(f"Error: {e}")
        return f"Error: Unable to get response from Gemini API. Details: {str(e)}"

# Streamlit input and button for user interaction
input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# Handle the submit button click
if submit and input:
    response = get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)

elif submit:
    st.warning("Please enter a question before submitting.")


# Display conversation history (skip the initial system prompt)
st.subheader("Conversation History:")
for i, msg in enumerate(st.session_state['flowmessages']):
    # Skip the first message which is the initial prompt
    if i == 0:
        continue
    st.write(f"**{msg['role'].capitalize()}:** {msg['parts'][0]['text']}")

