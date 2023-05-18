import sys

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

from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryBufferMemory
from langchain import OpenAI, PromptTemplate, LLMChain, ConversationChain
# from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']


template = """Assistant is a large language model trained by OpenAI. 
Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information 
on a wide range of topics. Whether you need help with a specific question or just want to have a conversation 
about a particular topic, Assistant is here to assist.
{history}
Human: {human_input}
Assistant:"""

DEFAULT_SYSTEM_PREFIX = ''''''

chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo",
                  openai_api_key=OPENAI_API_KEY,
                  )

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

prompt = PromptTemplate(
    input_variables=["history","human_input"],
    template=template
)

conversation = LLMChain(
    llm=chat,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=chat,  max_token_limit=1000, return_messages=True),
    prompt=prompt,
)




embeddings = None
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
CHROMA_ID_FULL_JSON = 'history'

if not (os.path.exists(os.path.join(DB_DIR, 'chroma-collections.parquet')) and os.path.exists(
        os.path.join(DB_DIR, 'chroma-embeddings.parquet'))):
    vectorstore = Chroma.from_texts(
        texts=[''],
        embedding=embeddings,
        persist_directory=DB_DIR)
    vectorstore.persist()
else:
    vectorstore = Chroma(
        # collection_name="langchain_store",
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()


#

def load_from_vectorstore():
    read_history = vectorstore._collection.get(ids=[CHROMA_ID_FULL_JSON], include=["documents"])
    print(read_history)
    if len(read_history['documents']) > 0 and len(read_history['documents'][0]) > 0:
        return json.loads(read_history['documents'][0])


def save_to_vectorstore(ids, text):
    vectorstore._collection.upsert(ids=[ids], documents=[json.dumps(text)])
    vectorstore.persist()

def load_history():
    history_load = load_from_vectorstore()
    if history_load:
        return history_load
    else:
        return ''


history_dict = load_history()
history_messages=messages_from_dict(history_dict)
# history_text = load_history()
history_text = str(history_dict)
print('history_text=', history_dict)
print('history_messages=', history_messages)

if history_messages:
    conversation.memory.chat_memory.messages=history_messages
    conversation.memory.predict_new_summary(history_messages, '')


print(conversation.memory.load_memory_variables({}))
print(conversation.memory.buffer)





@server.route("/", methods=("GET", "POST"))
def index():
    def primitive_formatter(message_list: list[dict]) -> str:
        result = ''
        for message in message_list:
            result += '<b>' + message["type"].capitalize() + ': </b>' + message["data"]["content"] + "<br>"
        return result
    global history_text, history_dict
    if request.method == "POST":
        human_text = request.form["text"]
        if human_text == 'clear history':
            history_text = ''
            conversation.memory.clear()
        else:
            print(f'sending req. hyman text={human_text}')
            response = conversation.predict(human_input=human_text, history='')
            print(f'response={response}')
        history_dict=messages_to_dict(conversation.memory.load_memory_variables({})["history"])
        print(
            f'conversation.memory.load_memory_variables()["history"]={history_dict}')
        history_text = str(history_dict)
        save_to_vectorstore(CHROMA_ID_FULL_JSON, history_dict)
        return redirect(url_for("index", last_response=response))

    # history = conversation.memory.load_memory_variables({})['history']
    # print(history_text)
    # print(load_from_vectorstore())
    get_last_response=request.args.get('last_response')
    if get_last_response and len(get_last_response)>0 and get_last_response!=history_dict[-1]["data"]["content"]:
        return render_template("index.html", result=primitive_formatter(history_dict+[{'type': 'ai', 'data': {'content': get_last_response}}]))

    return render_template("index.html", result=primitive_formatter(history_dict))



@server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    print('SHUTDOWN!')
    db_shutdown()
    sys.exit(0)

def db_shutdown():
    global vectorstore
    vectorstore.persist()
    vectorstore = None
