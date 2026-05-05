import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("🤖 FAQ Chatbot (Task 2)")
st.caption("Developed by Mahnoor Fatima | CodeAlpha AI Intern")

# Load FAQs
with open('faqs.json', 'r') as f:
    faq_data = json.load(f)

questions = list(faq_data.keys())

# User Input
user_question = st.text_input("Ask me something (e.g., What is Python?):")

if user_question:
    # NLP Logic: Cosine Similarity istemal karke best match dhoondna
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(questions + [user_question])
    
    # User question aur FAQ questions ke darmiyan similarity check karein
    cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    best_match_idx = cosine_sim.argsort()[0][-1]
    confidence = cosine_sim[0][best_match_idx]

    if confidence > 0.3:  # Agar 30% se ziada match ho
        st.success(f"**Answer:** {faq_data[questions[best_match_idx]]}")
    else:
        st.error("I'm sorry, I don't have an answer for that. Please try asking differently.")

# Sidebar for List of FAQs
with st.sidebar:
    st.header("Available Questions")
    for q in questions:
        st.write(f"- {q}")