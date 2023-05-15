
from flask import Flask, redirect, render_template, request, url_for

main = Flask(__name__)

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    Document,
    messages_from_dict, messages_to_dict
)
import json, os, sys

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



import chromadb
from chromadb.utils import embedding_functions


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
CHROMA_ID_FULL_JSON='CHROMA_ID_FULL_JSON'

client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR,
    anonymized_telemetry=False
)
embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )


vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    client_settings=client_settings,
    persist_directory=DB_DIR,
)

# vectorstore.persist()
# vectorstore.delete_collection()

print(vectorstore.get())
read_history = vectorstore._collection.get(ids=[CHROMA_ID_FULL_JSON],include=["documents"])
print(read_history)
if len(read_history['documents'])>0 and len(read_history['documents'][0])>0:
    conversation.memory = ConversationBufferMemory(
         chat_memory=ChatMessageHistory(messages=messages_from_dict(json.loads(read_history['documents'][0]))),
         return_messages=True)





@app.route("/", methods=("GET", "POST"))
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

        vectorstore._collection.upsert(ids=CHROMA_ID_FULL_JSON, documents=[history_json])
        vectorstore.persist()
        print(vectorstore._collection)
        print(vectorstore.get())

        return redirect(url_for("index", result='<br>\n'.join([message_formatted(message) for message in history])))

    history = conversation.memory.load_memory_variables({})['history']
    result = '<br>\n'.join([message_formatted(message) for message in history])

    print(vectorstore._collection)
    print(vectorstore.get())
    vectorstore.persist()
    return render_template("index.html", result=result)

# print()
#
# vectorstore.persist()

if __name__ == "__main__":
    try:
        app.run()
    except KeyboardInterrupt:
        print("Flask application is terminating...")
        vectorstore.persist()
        vectorstore = None
        # Add any cleanup code or necessary actions here
        # For example, closing database connections, saving data, etc.
        sys.exit()