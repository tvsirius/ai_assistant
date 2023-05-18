import sys

from flask import Flask, redirect, render_template, request, url_for

from werkzeug.serving import make_server

server = Flask(__name__)

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    Document,
    messages_from_dict, messages_to_dict
)
import json, os
import chromadb

from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryBufferMemory
from langchain import OpenAI, PromptTemplate, LLMChain, ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma



from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']

template= """Assistant is a large language model trained by OpenAI. 
Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information 
on a wide range of topics. Whether you need help with a specific question or just want to have a conversation 
about a particular topic, Assistant is here to assist.
{history}
Human: {human_input}
Assistant:"""

DEFAULT_SYSTEM_PREFIX=''''''

chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo",
                  openai_api_key=OPENAI_API_KEY,
                  )

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)

conversation = LLMChain(
    llm=chat,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=chat),
    prompt=prompt,
)


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-ada-002")

JSON_FILENAME='conversation.json'
def load_from_json():
    if os.path.exists(JSON_FILENAME):
        try:
            with open(JSON_FILENAME, 'r', encoding='utf-8') as filename:
                messages=json.loads(filename.read())
            if messages:
                return messages
        except:
            pass

def save_to_json(json):
    with open(JSON_FILENAME, 'w', encoding='utf-8') as filename:
        filename.write(json)


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
CHROMA_ID_FULL_JSON = 'CHROMA_ID_FULL_JSON'

if not (os.path.exists(os.path.join(DB_DIR,'chroma-collections.parquet')) and os.path.exists(os.path.join(DB_DIR,'chroma-embeddings.parquet'))):
    vectorstore = Chroma.from_texts(
        texts=[''],
        embedding=embeddings,
        persist_directory=DB_DIR)
    vectorstore.persist()
else:
    vectorstore=Chroma(
        # collection_name="langchain_store",
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=DB_DIR,
        )
    vectorstore.persist()

#

def load_from_vectorstore():
    """  ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    client_settings = chromadb.config.Settings(
        # chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    vectorstore = Chroma(
        # collection_name="langchain_store",
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=DB_DIR,
    )"""
    read_history = vectorstore._collection.get(ids=[CHROMA_ID_FULL_JSON],include=["documents"])
    print(read_history)
    if len(read_history['documents']) > 0 and len(read_history['documents'][0]) > 0:
        return read_history['documents'][0]

def save_to_vectorstore(ids,text):
    """    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    client_settings = chromadb.config.Settings(
        # chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        # anonymized_telemetry=False
    )

    vectorstore = Chroma(
        # collection_name="langchain_store",
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=DB_DIR,
    )"""
    doc=Document(page_content=text)
    vectorstore._collection.upsert(ids=[ids], documents=[text])
    # vectorstore.update_document(document_id=ids, document=doc)
    # vectorstore=Chroma.from_texts(ids=ids, texts=[text], persist_directory=DB_DIR, embedding=embeddings)
    vectorstore.persist()


history_load=load_from_vectorstore()
if history_load:
    history_text=history_load
else:
    history_text=''
print('history_text=', history_text)


@server.route("/", methods=("GET", "POST"))
def index():
    '''    def message_formatted(message: BaseMessage):
        if type(message) == type(AIMessage(content="")):
            text_from = 'Assistant: '
        elif type(message) == type(HumanMessage(content="")):
            text_from = 'Human: '
        else:
            text_from = 'System: '
        return '<b>' + text_from + '</b>' + message.content + '<br>'
    '''
    global history_text
    if request.method == "POST":
        human_text = request.form["text"]
        if human_text=='clear history':
            history_text=''
            conversation.memory.clear()
        elif human_text=='exit':
            exit(0)
        else:
            print(f'sending req. hyman text={human_text}, history_text={history_text}')
            response=conversation.predict(human_input=human_text, history=history_text)
            print(f'response={response}')

        print(f'conversation.memory.load_memory_variables()["history"]={conversation.memory.load_memory_variables({})["history"]}')

        history_text = conversation.memory.load_memory_variables({})['history']

        # print('-----\n\n--history_text = conversation.memory.load_memory_variables({})-\n\n---------')
        # print(history_text)

        save_to_vectorstore(CHROMA_ID_FULL_JSON,history_text)

        return redirect(url_for("index",))

    # history = conversation.memory.load_memory_variables({})['history']
    # print(history_text)
    print(load_from_vectorstore())
    # result = '<br>\n'.join([message_formatted(message) for message in history])

    return render_template("index.html", result=history_text)


def shutdown_server():
    print('SHUTDOWN!')
    global vectorstore,chat,prompt,conversation,server
    vectorstore.persist()
    chat=None
    prompt=None
    conversation=None
    server=None
    vectorstore=None
    # func = request.environ.get('werkzeug.server.shutdown')
    # if func is None:
    #     raise RuntimeError('Not running with the Werkzeug server')
    # func()

@server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def db_shutdown():
    print('SHUTDOWN!')
    global vectorstore,chat,prompt,conversation,server
    vectorstore.persist()
    chat=None
    prompt=None
    conversation=None
    server=None
    vectorstore=None