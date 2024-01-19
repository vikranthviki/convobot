import streamlit as st
from convo import qa

st.title("Conversational QA App")

# Get user input
user_query = st.text_input("Ask a question:")

if st.button("Ask"):
    # Perform the QA and display the result
    result = qa({"question": user_query, "chat_history": []})
    st.write("Answer:", result["result"])