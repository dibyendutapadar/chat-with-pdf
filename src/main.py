import os
import streamlit as st
from doc_chat_utility import get_answer


working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title= "Software usage guide",
    layout="centered"
)

st.title("Document QA  - llama3 - Ollama")

uploaded_file = st.file_uploader(label="Upload your file",type=['pdf'])
user_query = st.text_input(label="What you want to know")

if st.button("Search"):
    bytes_data=uploaded_file.read()
    file_name = uploaded_file.name

    # save the file to the working directory
    file_path = os.path.join(working_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(bytes_data)
    answer = get_answer(file_name,user_query)

    st.success(answer)