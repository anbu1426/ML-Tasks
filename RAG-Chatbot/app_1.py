# Importing the neccessary libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# to make the input into chunks
from langchain_text_splitters import CharacterTextSplitter 

# to embed the data and to send to the vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings 

# used for vectorstore
from langchain_community.vectorstores import FAISS 

# This is ollama we need to download ollama model from net
from langchain_ollama import ChatOllama 

from htmlTemplate import css, user_template, bot_template

from langchain_core.chat_history import BaseChatMessageHistory

# So chatprompt template converts the chat into a well structured prompt with defining who said what? and the msg placeholder is a slot that gets filled with actual chathistory at runtime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

# This is a key wrapper that makes chain-memory aware. This intercepts at each call, checks whether the history is right or wrong then if new data is available then it adds it into the history and saves automatically
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# Function to get raw text(this takes the pdfs and returns a single string full of texts from the pdfs given)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:# To take one-by-one pdf from a bunch
        pdf_reader = PdfReader(pdf) # Using this class we can get the data of that one pdf in that
        for page in pdf_reader.pages: # So that as an attribute - page, so page-by-page we are extracting the data
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text # Returning the raw string 

# Function to make the raw text into text chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,# Each chunk has 1000 chars
        chunk_overlap=200,# Useful when we come to start part of the para and chunk_size of 1000 chars does not satisfy.So according to this parameter we can take the 200 chars from the before text which is already present in another chunk no issues
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store with text chunks converted to embeddings
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Can use embedding fn from instructors too. But for me this works....

    vectorstore = FAISS.from_texts(texts=chunks,
                                   embedding=embeddings)
    return vectorstore
# A special fn happens when these chunks are loaded to the vectorstore which makes the comparsion of the chunks with question vector much easier

# Function to do conversation chain
def build_rag_chain(vectorstore):
    # Actual ai that is going to answer(llm brain)
    llm = ChatOllama(model='phi3', # Can download llama3 also(but it is 5gb)
                     temperature=0) # This make the decision determistic(zero randomness)
    
    # So instead of matching the exact words it finds the most similar chunks related to the question   
    retriever = vectorstore.as_retriever(search_kwargs={'k':4}) # retrieve 4 closest chunk to the question

    # Rephrases the question, creates the context
    # Eg, You -> Who is the author?
    #     Bot -> The author is Shiv Khera
    #     You -> Ok. How old is he? So now the retriever does not know who is 'he'(lacks the context)

    # This fn takes the current question + the entire chat history to solve the above problem
    # Same eg, You -> Who is the author?
    #          Bot -> The author is Shiv Khera
    #          You -> Ok. How old is he?
    # Now this fn rephares it as follows,
    # Chat history = The author is Shiv Khera
    # Now the current question repharses to Ok. How old is he? -> ok. How old is Shiv Khera?
    contextualize_ques_prompt = ChatPromptTemplate.from_messages([
        ("ai","...rewrite the question to be a self-contained query..."), # ai instruction tells the llm to rewrite the question from taking the chat history that is injected in the next line
        MessagesPlaceholder("chat_history"), # Injecting the chat history so far
        ("human","{input}") # Takes the current question
    ])

    # So what does this do then?
    # Step 0 -> Our question 
    # Step 1 -> llm reads the chat history + current question through the above class/fn
    # Step 2 -> Repharses the question with context then the retreiver searchs for this question in vectorstore and find 4 most relevant chunks to this question
    # And then returns the 4 relevant chunks to the answer chain
    # Without this the retriever can't find any relevant answers    
    # Without this the chatbot would not be memory-aware i.e., forgets the previous convo
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_ques_prompt
    )

    # Actually answering the question
    # ai instruction -> follow the rules and context(tells the llm to only find the answers from these chunk)
    # chat history -> makes the llm remember all the previous convo
    # user instruction -> give the current question
    ques_ans_prompt = ChatPromptTemplate.from_messages([
        ("ai", "...answer strictly based on the context...\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    # So this stuffs the 4 relevant chunks to the context slot 
    # And sends to the llm and generates the answer
    ques_ans_chain = create_stuff_documents_chain(llm, ques_ans_prompt)

    # This combines the two halves -> Part_1(rephrase the question + return the 4 relevant chunks) + Part_2(stuff the relevant 4 chunks + then generate the ans)
    rag_chain = create_retrieval_chain(history_aware_retriever, ques_ans_chain)
    return rag_chain

# Creating a function is just the memory manager of the model
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # When the model runs for the first time there is no chat history so we create one initializing with empty dict
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    
    # So if the current session of the chat is not in the chat history initializing it.
    # ChatMessageHistory just creates the list of the Messages
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

# Handling the question
# What this does?
# rag_chain(no memory) -> RunnableWithMessageHistory(makes the rag chain memory aware) -> conversational_rag(memory aware)
def handle_user_input(user_question):
    conversational_rag = RunnableWithMessageHistory(
        st.session_state.rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input", # The input goes under the key name "input"
        history_messages_key="chat_history", # The chat history goes here
        output_messages_key="answer" # The answer goes here
    )

    # Responding to the question
    response = conversational_rag.invoke(
        # The question goes here
        {"input":user_question},

        # This session_id = main_session then runs the get_session_history and then gets the chat history for that session to contextualize and answer the question
         config={"configurable": {"session_id": "main_session"}}
    )
    answer = response["answer"]

    # Short summary on how this worked(the fns we used works like this together to ans our questions)
    # Step-0 -> User ques
    # Step-1 -> Finding the session_id => Fetching that session's chat history
    # Step-2 -> history_aware_retriever => repharses + give 4 relevant chunks
    # Step-3 -> ques_ans_chain => generates the answer
    # Step-4 -> RunnableWithChatHistory => Saves the Q&A with the history
    # Step-5 -> Gives the response

    # Append to display history
    st.session_state.display_history.append(("user", user_question))
    st.session_state.display_history.append(("bot", answer))

def main():
    load_dotenv() # This is done so that the variables in the .env file are securely loaded with the system variables(project's environment variables)
    st.set_page_config(page_title='AskBot', page_icon=":books:")
    st.write(css, unsafe_allow_html=True) # That boolean is to say to streamlit that it must parse through the html file given

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "display_history" not in st.session_state:
        st.session_state.display_history = None


    st.header("AskBot :books:")
    user_question = st.text_input("Ask any question here....")

    if user_question:
        handle_user_input(user_question)

    # Rendering the chat(newest at top) and displaying the convo b/w the bot and user
    for role, msg in reversed(st.session_state.display_history or []):
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else: 
            st.write(bot_template.replace("{{MSG}}",msg), unsafe_allow_html=True)

    # Setting the sidebar
    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader("Upload the pdf's here....",
                                    accept_multiple_files=True)
        if st.button("Process Docs"):
            if not pdf_docs:
                st.warning("Atleast upload one pdf to chat about it")
            else:
                with st.spinner("Processing...."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                       
                    # Create a instance for the fn(session_state keeps the variable persistent when the streamlit automatically reloads it up then the variables are re-intialized this prevents from it)
                    st.session_state.rag_chain = build_rag_chain(vectorstore)

                    # Clear the old chat when new docs are loaded
                    st.session_state.display_history = []
                    if "chat_histories" in st.session_state:
                        del st.session_state.chat_histories
                st.success(f"Processed {len(text_chunks)} from {len(pdf_docs)} PDF(s)")

        if st.button("Clear Chat"):
            st.session_state.display_history = []
            if "chat_histories" in st.session_state:
                del st.session_state.chat_histories
            st.rerun() # To reload the page


if __name__ == "__main__":
    main()