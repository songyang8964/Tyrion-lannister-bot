# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from PIL import Image
from pinecone import Pinecone
from decouple import config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import coloredlogs, logging
import os
from constants import (
    DEFAULT_CHAT_MODEL,
    EMBEDDING_MODEL
)


# --------------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))
st.set_page_config(page_title='Game of Thrones Chatbot', page_icon='⚔️', initial_sidebar_state="auto", menu_items=None)
st.title("Tyrion Lannister - Game of Thrones ⚔️")

tyrion_image = Image.open("Tyrion.jpg")  
st.image(tyrion_image, caption="Tyrion Lannister", use_column_width=True)


st.markdown("""
**Tyrion Lannister**  
Hello, I am Tyrion, also known as "The Imp" or "The Halfman".  
A sharp wit, an even sharper tongue, and a love for wine and books are my trademarks.  
In a world of swords and politics, I survive with my intellect. Shall we chat?
""")

st.sidebar.subheader("Enter Your API Key 🗝️")
open_api_key = st.sidebar.text_input(
    "Open API Key", 
    value=st.session_state.get('open_api_key', ''),
    help="Get your API key from https://openai.com/",
    type='password'
)

open_api_key = os.environ.get("OPENAI_API_KEY")
st.session_state['open_api_key'] = open_api_key
# load_dotenv(find_dotenv())

with st.sidebar.expander('Advanced Settings ⚙️', expanded=False):
    open_ai_model = st.text_input('OpenAI Chat Model', DEFAULT_CHAT_MODEL, help='See model options here: https://platform.openai.com/docs/models/overview')   


def get_vectorstore():
    # text_field = "text"
    embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "got-bot"
    index = pc.Index(index_name)
    vectorstore =  PineconeVectorStore(
        index=index, 
        embedding=embed
    )

    return vectorstore

@st.cache_data
def get_response_from_question(_vectorstore, question, memory, k=10):

    docs = _vectorstore.similarity_search(question, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name=open_ai_model, temperature=0)

    # Template to use for the system message prompt
    template = """
        Your name is Tyrion Lannister, son of Tywin Lannister.

        You are an expert on the history of Westeros, and seasoned in the art of war and politics.
        
        Here is relevant information on the history of Westeros to 
        help you answer the following questions: {docs}

        Here is relevant information from the current conversation to help you answer the following question: {memory}
        
        Only use the factual information from these books to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "Sorry, I'm a Dwarf not a wizard, I don't know the answer to that".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = """
        Someone is coming to you (Tyrion Lannister) for your expert advice, they are in need of your help.
        Here is there question: {question}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=question, docs=docs_page_content, memory=memory)
    return response, docs

if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

question = st.text_input(
    label="Ask Tyrion a question",
    value="What role did you play in the battle of the blackwater?"
)

if question != "" and (open_api_key == '' or open_api_key is None):
    st.error("⚠️ Please enter your API key in the sidebar")
else:
    vectorstore = get_vectorstore()
    if len(st.session_state['questions']) > 0:
        memory = '\n\n'.join(
            [
                f'Question: {q}\nAnswer: {a}'
                for q, a in zip(st.session_state['questions'], st.session_state['responses'])
            ]
        )
    else:
        memory = None
    response, docs = get_response_from_question(vectorstore, question=question, memory=memory, k=10)

    st.session_state['questions'].append(question)
    st.session_state['responses'].append(response)

    for i in range(len(st.session_state['questions'])):
        question = st.session_state['questions'][i]
        response = st.session_state['responses'][i]
        message(question, is_user=True)  # align's the message to the left
        message(response, is_user=False)  # align's the message to the right

