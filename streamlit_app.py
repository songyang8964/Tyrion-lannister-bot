from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from PIL import Image
from pinecone import Pinecone
from decouple import config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import coloredlogs, logging
import os
from constants import (
    DEFAULT_CHAT_MODEL,
    EMBEDDING_MODEL
)
from langchain.memory import ConversationBufferMemory

# --------------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))
st.set_page_config(page_title='Game of Thrones Chatbot', page_icon='⚔️', initial_sidebar_state="auto", menu_items=None)
st.title("Tyrion Lannister - Game of Thrones ")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question" not in st.session_state:
    st.session_state.question = ""

tyrion_image = Image.open("Tyrion.jpg")  
st.image(tyrion_image, caption="Tyrion Lannister", use_column_width=True)

st.markdown("""
**Tyrion Lannister**  
Hello, I am Tyrion, also known as "The Imp" or "The Halfman".  
A sharp wit, an even sharper tongue, and a love for wine and books are my trademarks.  
In a world of swords and politics, I survive with my intellect. Shall we chat?
""")

st.markdown("""
**Example questions you can ask:**
- What happened at the Battle of Blackwater Bay?
- Tell me about your relationship with your family.
- What do you think about dragons?
- How did you become Hand of the King?
""")

# Input area right after example questions
cols = st.columns([0.85, 0.15])
with cols[0]:
    question = st.text_input(
        "Ask Tyrion a question",
        key="question",
        value=st.session_state.question,
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
with cols[1]:
    submit_button = st.button("Ask", type="primary", use_container_width=True)

st.sidebar.subheader("Enter Your API Key ")
open_api_key = st.sidebar.text_input(
    "Open API Key", 
    value=st.session_state.get('open_api_key', ''),
    help="Get your API key from https://openai.com/",
    type='password'
)

open_api_key = os.environ.get("OPENAI_API_KEY")
st.session_state['open_api_key'] = open_api_key

# Initialize memory in session state if it doesn't exist
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

def get_vectorstore():
    embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "got-bot"
    index = pc.Index(index_name)
    vectorstore =  PineconeVectorStore(
        index=index, 
        embedding=embed
    )
    return vectorstore

def get_response_from_question(_vectorstore, question, k=10):
    docs = _vectorstore.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name=open_ai_model, temperature=0.7)

    # Template to use for the system message prompt
    template = """
        You are Tyrion Lannister, the witty and clever son of Tywin Lannister. Known as 'The Imp' or 'Halfman', you've learned to use your wit as your weapon.
        
        Key personality traits to incorporate in your responses:
        - Highly intelligent and well-read, often quoting books and sharing wisdom
        - Sharp wit and clever humor, especially in tense situations
        - Cynical yet insightful view of the world
        - Love for wine and worldly pleasures, which you often reference
        - Self-deprecating humor about your height and appearance
        - Deep understanding of politics and human nature
        - Complicated relationship with your family, especially your father and sister
        
        Speaking style:
        - Use clever wordplay and witty remarks
        - When appropriate, briefly reference your memorable quotes from past conversations
        - Occasionally make references to wine or drinking
        - Include subtle sarcasm and irony
        - Mix wisdom with humor in your responses
        - When appropriate, reference your own experiences and struggles
        - If the context is right, you may briefly quote your own past words that are relevant to the current topic
        
        Some of your memorable quotes (use sparingly and only when truly relevant):
        - "A mind needs books like a sword needs a whetstone."
        - "I have a tender spot in my heart for cripples, bastards, and broken things."
        - "Never forget what you are. The rest of the world will not. Wear it like armor, and it can never be used to hurt you."
        - "That's what I do: I drink and I know things."
        - "Death is so final, whereas life is full of possibilities."
        - "It's not easy being drunk all the time. Everyone would do it if it were easy."
        - "I try to know as many people as I can. You never know which one you'll need."
        
        Background knowledge:
        - You are an expert on the history of Westeros
        - Former Hand of the King to both Joffrey and Daenerys
        - Seasoned in warfare (Battle of Blackwater) and politics
        - Well-traveled across both Westeros and Essos
        
        Here is relevant information from the books to help you answer the question: {docs}
        
        Remember to:
        1. Stay true to your character while using the factual information provided
        2. You may include witty observations or clever jests
        3. When appropriate and the context is right, you can briefly quote your own past words
        4. Ensure your core answer is based on the provided context
        5. Keep any quotes very brief and relevant to the current topic
        6. Don't force quotes - only use them when they naturally fit the conversation
        7. Reference previous conversations when relevant
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "A curious soul seeks your counsel, Lord Tyrion. They ask: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        human_message_prompt
    ])

    chain = LLMChain(
        llm=chat,
        prompt=chat_prompt,
        memory=st.session_state.memory,
        verbose=True
    )

    response = chain({"question": question, "docs": docs_page_content})
    return response["text"], docs

# Advanced settings in sidebar
st.sidebar.subheader('Advanced Settings')
with st.sidebar.expander('Model Settings', expanded=False):
    open_ai_model = st.text_input('OpenAI Chat Model', DEFAULT_CHAT_MODEL, help='See model options here: https://platform.openai.com/docs/models/overview')   

# Load Tyrion's avatar image once
tyrion_avatar = Image.open("Tyrion.jpg")
# Resize the avatar to a smaller size suitable for chat
tyrion_avatar = tyrion_avatar.resize((40, 40))

def clear_input():
    st.session_state.question = ""

# Display chat messages in a container
chat_container = st.container()
with chat_container:
    for message_obj in st.session_state.messages:
        if message_obj["role"] == "user":
            message(message_obj["content"], is_user=True, avatar_style="personas")
        else:
            with st.container():
                cols = st.columns([1, 15])
                with cols[0]:
                    st.image(tyrion_avatar, width=40)
                with cols[1]:
                    st.markdown(f"<div style='background-color: #2e2e2e; padding: 12px; border-radius: 10px; margin-bottom: 10px;'>{message_obj['content']}</div>", unsafe_allow_html=True)

# Handle the conversation
if (question and submit_button) and (open_api_key == '' or open_api_key is None):
    st.error(" Please enter your API key in the sidebar")
elif question and submit_button:
    vectorstore = get_vectorstore()
    response, docs = get_response_from_question(vectorstore, question=question, k=10)

    # Store messages for display
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the display
    st.experimental_rerun()
