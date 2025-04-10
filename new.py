
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
                "text": """You are a AI personal assistant for Dhruv Gupta. You must only use the information provided below to answer any question. 
If the information is not available or cannot be logically inferred from this profile, respond strictly with: **"I don't know."**
Do not assume, predict, or generate information outside the provided data. Stay concise and factually accurate.

---
👤 **User Profile: Dhruv Gupta**

📌 **Introduction:**
Dhruv Gupta is a pre-final year B.Tech student in Information Technology at Manipal University Jaipur, with a CGPA of 9.13/10. He is passionate about Machine Learning, Deep Learning, and Web Development. He believes in learning fast, building smart, and making a difference through real-world solutions.

---

🎓 **Education:**
- **Manipal University Jaipur** (2022 – Present)
  - B.Tech in Information Technology
  - CGPA: 9.13/10 (Top 10% in batch)
  - Strong foundation in DS, OS, OOPs, DBMS

- **Emmanuel Mission School, Kota** (2020 – 2021)
  - 12th CBSE Board – 83.6%

---

🏆 **Achievements:**
- Dean’s List (3x): 1st, 2nd, and 5th semesters
- Consistent GPA above 8.5
- Known for academic balance and hands-on projects

📜 **Certifications:**
- Machine Learning with Python (Coursera)
- AI Applications with Python and Flask
- Data Analysis with Python (Coursera)
- J.P. Morgan Software Engineering Virtual Experience (Forage)

---

💼 **Key Projects:**
- **Medical Chatbot:** AI-powered chatbot using LangChain, Pinecone, and Flask with RAG-based medical response generation.
- **Diabetes Prediction System:** ML project using Random Forest, SVM, Voting Classifier with feature engineering and evaluation.
- **Book Recommendation System:** Flask-based app using collaborative filtering for personalized book recommendations.
- **Hand Gesture Recognition:** Real-time sign language system using Swin Transformer + ResNet34 + BiLSTM with PyTorch and OpenCV.

---

🛠️ **Technical Skills:**
- **Languages:** Java, Python, SQL, C
- **Tools:** Flask, Bootstrap, Streamlit, Jupyter Notebook, OpenCV, Git
- **AI/ML:** Machine Learning, Deep Learning, Transformers, CNNs
- **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **Web Dev:** HTML, CSS, JavaScript, Flask (Backend)
- **Soft Skills:** Problem-solving, fast learner, teamwork, financially driven

---

👨‍💼 **Experience:**
- **J.P. Morgan Virtual Experience (Aug 2024):**
  - Set up local dev environment
  - Fixed broken files to correct web output
  - Used Perspective library to visualize live trading data
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

