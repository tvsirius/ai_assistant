from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

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

from langchain.memory import ConversationBufferMemory,ChatMessageHistory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain, VectorDBQA
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator


from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']

chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo",
                  openai_api_key=OPENAI_API_KEY,
                  )

conversation = ConversationChain(
    llm=chat,
    verbose=True,
    memory=ConversationBufferMemory(return_messages=True)
)



import chromadb
from chromadb.utils import embedding_functions

# chroma_client = chromadb.Client()
# from chromadb.config import Settings
# client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="convdb/" # Optional, defaults to .chromadb/ in the current directory
# ))
# # client.reset()


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
CHROMA_ID_FULL_JSON='CHROMA_ID_FULL_JSON'


client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR,
    anonymized_telemetry=False
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,)

vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    client_settings=client_settings,
    persist_directory=DB_DIR,
)




vectorstore.persist()
# vectorstore.delete_collection()


index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),

)

print(vectorstore._collection)
read_history = vectorstore._collection.get(ids=[CHROMA_ID_FULL_JSON],include=["documents"])
print(read_history)
if len(read_history['documents'])>0 and len(read_history['documents'][0])>0:
    conversation.memory = ConversationBufferMemory(
         chat_memory=ChatMessageHistory(messages=messages_from_dict(json.loads(read_history['documents'][0]))),
         return_messages=True)


# print(read_history)
#
# if len(read_history['documents']) > 0 and len(read_history['documents'][0]) > 0:
#     conversation.memory = ConversationBufferMemory(
#         chat_memory=ChatMessageHistory(messages=messages_from_dict(json.loads(read_history['documents'][0]))),
#         return_messages=True)

@app.route("/", methods=("GET", "POST"))
def index():
    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

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
        else:
            response=conversation.run(input=human_text)
            # print(response)
            # print(conversation.memory)

        history = conversation.memory.load_memory_variables({})['history']

        history_json=json.dumps(messages_to_dict(history))
        # messages_ret=messages_from_dict(json.loads(history_json))
        vectorstore._collection.upsert(ids=CHROMA_ID_FULL_JSON, documents=[history_json])
        vectorstore.persist()
        print(vectorstore._collection)
        vectorstore = None

        # read_history=collection.get(ids=[CHROMA_ID+'FULL_JSON'],include=["documents"])
        # print(json.loads(read_history['documents'][0]))
        # conversation.memory=ConversationBufferMemory(chat_memory=ChatMessageHistory(messages=messages_from_dict(json.loads(read_history['documents'][0]))), return_messages=True)


        return redirect(url_for("index", result='<br>\n'.join([message_formatted(message) for message in history])))

    history = conversation.memory.load_memory_variables({})['history']
    result = '<br>\n'.join([message_formatted(message) for message in history])
    #else:
    # result = request.args.get("result")
        # result=''
    print(vectorstore._collection)
    return render_template("index.html", result=result)

print()#



