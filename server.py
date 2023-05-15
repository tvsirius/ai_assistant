from flask import Flask, redirect, render_template, request, url_for

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

from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain import OpenAI, PromptTemplate, LLMChain

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

chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo",
                  openai_api_key=OPENAI_API_KEY,
                  )

# embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)

conversation = LLMChain(
    llm=chat,
    verbose=True,
    memory=ConversationBufferMemory(return_messages=True),
    prompt=prompt,
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-ada-002")

CHROMA_ID_FULL_JSON = 'CHROMA_ID_FULL_JSON'

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

def load_from_vectorstore():

    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    read_history = vectorstore._collection.get(ids=[CHROMA_ID_FULL_JSON],include=["documents"])
    print(read_history)
    vectorstore=None
    client_settings=None
    if len(read_history['documents']) > 0 and len(read_history['documents'][0]) > 0:
        return messages_from_dict(json.loads(read_history['documents'][0]))
    chromadb.Client().reset()

def save_to_vectorstore(ids,json):
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    vectorstore.add_texts(texts=[json], ids=ids)
    vectorstore.persist()
    vectorstore=None
    client_settings=None
    chromadb.Client().reset()



@server.before_first_request
def load_collection():
    get_messages=load_from_json()
    if get_messages:
        conversation.memory = ConversationBufferMemory( chat_memory=ChatMessageHistory(messages=messages_from_dict(get_messages)), return_messages=True)



@server.route("/", methods=("GET", "POST"))
def index():

    def message_formatted(message: BaseMessage):
        if type(message) == type(AIMessage(content="")):
            text_from = 'Assistant: '
        elif type(message) == type(HumanMessage(content="")):
            text_from = 'Human: '
        else:
            text_from = 'System: '
        return '<b>' + text_from + '</b>' + message.content + '<br>'


    if request.method == "POST":
        human_text = request.form["text"]
        if human_text=='clear history':
            conversation.memory=ConversationBufferMemory(return_messages=True)
        elif human_text=='exit':
            exit(0)
        else:
            response=conversation.predict(human_input=human_text)

        history = conversation.memory.load_memory_variables({})['history']

        history_json=json.dumps(messages_to_dict(history))

        # save_to_vectorstore(CHROMA_ID_FULL_JSON,history_json)

        save_to_json(history_json)

        return redirect(url_for("index",))

    history = conversation.memory.load_memory_variables({})['history']
    print(history)
    result = '<br>\n'.join([message_formatted(message) for message in history])

    return render_template("index.html", result=result)


"""
def server_shutdown():
    print("Application is terminating...")
    global vectorstore
    if vectorstore: vectorstore.persist()
    # print(vectorstore.persist())
    # print("Vectorstore persisted...")
    # print(vectorstore._client.persist())
    # print("Vectorstore client persisted...")
    vectorstore = None
    print("Vectorstore set none...")
#
"""