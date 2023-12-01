import streamlit as st
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


#Convierte los Pdfs en chunks de texto
def get_text_chunks(docss):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(docss)
    return chunks

#Crea la base de vectores
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

#Crea la cadena de conversación
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4', temperature=1)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3}),
        memory=memory
    )
    return conversation_chain

#Maneja la entrada del usuario
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


#Función principal
def main():
    load_dotenv()
    st.set_page_config(page_title="Chokobot",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chokobot :robot_face::seedling:")
    user_question = st.text_input("¿Cómo puedo ayudarte hoy?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Pulsa 'PROCESAR' para tener la información mas actualizada")
        
        # Cargar los Pdfs
        loader = PyPDFDirectoryLoader("pdfs/")
        docss = loader.load()

        # Procesar los Pdfs
        if st.button("Procesar"):
            with st.spinner("Procesando..."):

                # Pasar los Pdf a chunks
                text_chunks = get_text_chunks(docss)

                # Crear la base de Vectores
                vectorstore = get_vectorstore(text_chunks)

                # Crear la cadena(chain) de Conversación
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("¡Listo!")

if __name__ == '__main__':
    main()
