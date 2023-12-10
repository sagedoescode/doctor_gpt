#coding part
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64


#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('gradiant.jpeg')  

#sidebar contents

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#sidebar contents
with st.sidebar:
    st.title('Recent chats')
    st.markdown('''
    History:

    ''')

    # Display chat history in sidebar with clickable buttons
    for i, chat in enumerate(st.session_state['chat_history']):
        if st.button(chat, key=f'chat_{i}'):
            st.write(f"You clicked on chat: {chat}")

    add_vertical_space(4)
    st.write('🩺Health pdf based chatbot, designed by S.🤗')

load_dotenv()

def main():
    st.header("📄Chat with your medical test👨‍⚕️")

    #upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        st.write(pdf.name)
    else:
        st.write("Please upload a PDF file to chat with.")


    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()

            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        # Define a template for the QA chain prompt
        template = """You're an AI that know everything about medicine and you behave like a doctor.
        Helpful Answer:"""
        QA_CHAIN_PROMPT = SystemMessagePromptTemplate.from_template(template)

        # Get the user's query
        query = st.text_input("Ask questions about the uploaded PDF file")

        # Check if the submit button was clicked
        submit_button = st.button("Submit Query 🤖")
        if submit_button:
            # Add the query to the chat history
            st.session_state["chat_history"].append(query)

            # Find the most relevant documents to the query using FAISS
            docs = vectorstore.similarity_search(query=query, k=3)

            # Use the Chain of Thought model to generate a response
            llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Generate the response using the Chain of Thought model
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            # Display the response
            st.write(f"**Doctor:** {response}")



if __name__=="__main__":
    main()
