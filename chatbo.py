import streamlit as st
from dotenv import load_dotenv # load .env into os.environ
import os
from langchain_groq import ChatGroq # groq llm integration
from langchain.memory import ConversationBufferMemory # memory backend for chat
from langchain.chains import ConversationChain # it wires LLM + memory

# load api key

# load_dotenv() # read .env file

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



st.set_page_config(page_title=" 💬 Conversational Chatbot") # title in browser tab

# Apply theme CSS
a = st.sidebar.selectbox("Choose your theme", ["Dark", "Colourful"])
if a == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stTextInput, .stSelectbox, .stSlider {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    .stTitle {
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)
elif a == "Colourful":
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#000000 50%, #800080 100%);
        colour:#8F00FF;
    }
    .stSidebar {
        background: linear-gradient(135deg, #000000 50%, #800080 100%);
        color: #FFB6C1;
    }
    .stTextInput, .stSelectbox, .stSlider {
        background-color: #000000 !important;
        color: #000000 !important;
        border-radius: 8px;
    }
    .stTitle {
        color: #000000 ;
        font-weight: bold;
    }
    
    
    </style>
    """, unsafe_allow_html=True)

# st.title("💬 Conversational Chatbot with Message History") # app header
st.title("Conversational Chatbot with Message History")
# sidebar control
key=st.sidebar.text_input("Enter your Groq API key ",type="password")


model_name = st.sidebar.selectbox( 
    "Select Groq Model",
    ["openai/gpt-oss-120b","meta-llama/llama-4-maverick-17b-128e-instruct","qwen/qwen3-32b"]
    )

temperature = st.sidebar.slider( # fix the randomness of the response
    "Temperture",0.0,1.0,0.7
)

max_tokens = st.sidebar.slider( # max response length
    "Max Tokens", 50,1500,150
)




if "memory" not in st.session_state:
    # perssist memory across reruns
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True # return as list of memory, not in one big string.
    )

if "history" not in st.session_state:
    st.session_state.history=[]

# user input

user_input = st.chat_input("Ask Questions") # clears itself on enter

if user_input:
    st.session_state.history.append(("user", user_input))

    # initialized a fresh llm for this turn
    llm= ChatGroq(
        api_key=key,
        model_name=model_name,
        temperature= temperature,
        max_tokens= max_tokens
    )

    # build conversation chain in our memory
    conv =  ConversationChain(
        llm=llm,
        memory= st.session_state.memory,
        verbose= True
    )

    ## get ai response ( memory is updted internally)
    ai_response = conv.predict(input=user_input)

    # append assitant to history
    st.session_state.history.append(("assistant",ai_response))

# render  chat bubble
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text) # user style
    else:
        st.chat_message("assistant").write(text) # assistant style






