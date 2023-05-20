import sys, os

from flask import Flask, redirect, render_template, request, url_for

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage, Document, \
    messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryBufferMemory
from langchain import OpenAI, PromptTemplate, LLMChain, ConversationChain

# import whisper
import openai

# import pyttsx3


server = Flask(__name__)

from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']

from vectorstore import vectorstore, load_history, load_from_vectorstore, save_to_vectorstore

from ai_roles import ai_init_string

CHROMA_ID_FULL_JSON = 'history'


def history_formatter(message_list: list[dict]) -> str:
    result = ''
    if message_list and len(message_list) > 0:
        for message in message_list:
            result += message["type"] + ': ' + message["data"]["content"] + "\n"
    return result


class CustomPromt(PromptTemplate):
    def format(self, **kwargs) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        kwargs["history"] = history_formatter(messages_to_dict(kwargs["history"]))
        return super().format(**kwargs)


template = """{ai_init_string}
{history}
Human: {human_input}
Assistant:"""

AI_ASSISTANT_ROLE = "default"

prompt = CustomPromt(
    input_variables=["history", "human_input"],
    template=template,
    partial_variables={"ai_init_string": ai_init_string[AI_ASSISTANT_ROLE]['prompt_intro']}
)

chat = ChatOpenAI(temperature=ai_init_string[AI_ASSISTANT_ROLE]['temperature'], model_name="gpt-3.5-turbo",
                  openai_api_key=OPENAI_API_KEY,
                  )

conversation = LLMChain(
    llm=chat,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=chat, max_token_limit=2000, return_messages=True),
    prompt=prompt,
)

history_dict = load_history(CHROMA_ID_FULL_JSON)

if history_dict and len(history_dict) > 0:
    history_text = history_formatter(history_dict)
    history_messages = messages_from_dict(history_dict)

    if history_messages:
        conversation.memory.chat_memory.messages = history_messages
        # conversation.memory.predict_new_summary(history_messages, '')

print(conversation.memory.load_memory_variables({}))

audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recording.webm')


def chat_process_input(human_text):
    global history_dict, history_text
    if human_text == 'clear history':
        history_text = ''
        history_dict = []
        conversation.memory.clear()
        print(f'conversation.memory history={history_dict}')
        save_to_vectorstore(CHROMA_ID_FULL_JSON, history_dict)
        return 'conversation history cleaned'
    else:
        print(f'sending req. human text={human_text}')
        response_text = conversation.predict(human_input=human_text)
        print(f'response={response_text}')
    history_dict = messages_to_dict(conversation.memory.load_memory_variables({})["history"])
    print(f'conversation.memory history={history_dict}')
    history_text = history_formatter(history_dict)
    save_to_vectorstore(CHROMA_ID_FULL_JSON, history_dict)
    return response_text


@server.route('/record', methods=['POST', "GET"])
def record():
    print(" AUDIO REQUEST ")

    file = request.files['audio']
    file.save(audio_file)

    if os.path.exists(audio_file):
        with open(audio_file, 'rb') as file:
            transcript = openai.Audio.transcribe("whisper-1", file, prompt=history_text)
            # print(transcript)
            human_text = transcript['text']
        os.remove(audio_file)
    else:
        human_text = ''

    print('Human text = ', human_text)

    if len(human_text) > 0:
        response_text = chat_process_input(human_text)
    else:
        response_text = ''
    print('POST /record done')
    # print(history_dict)
    print(f'Sending response "input": {human_text}, "output": {response_text}')
    # return redirect(url_for("index", ))
    return {"input": human_text, "output": response_text}


@server.route("/", methods=("GET", "POST"))
def index():
    def html_formatter(message_list: list[dict]) -> str:
        result = ''
        if len(message_list) > 0:
            for message in message_list:
                result += '<b>' + message["type"].capitalize() + ': </b>' + message["data"]["content"] + "<br>"
        return result

    if request.method == "POST":
        print(" / POST REQUEST ")
        human_text = request.form["text"]
        if len(human_text) > 0:
            response_text = chat_process_input(human_text)
        else:
            response_text = ''

        return redirect(url_for("index", ))  # last_response=response_text

    print(" / GET REQUEST ")
    # get_last_response = request.args.get('last_response')
    # print('get_last_response=',get_last_response)
    if history_dict and len(history_dict) > 0:
        # if get_last_response and len(get_last_response) > 0 and get_last_response != history_dict[-1]["data"]["content"]:
        #     result_str = html_formatter(history_dict + [{'type': 'ai', 'data': {'content': get_last_response}}])
        # else:
        result_str = html_formatter(history_dict)
    # elif get_last_response=='conversation history cleaned':
    #         result_str='<b>System: </b> Conversation history cleaned <br>'
    else:
        result_str = ''
    return render_template("index.html", result=result_str, )


@server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def shutdown_server():
    print('SHUTDOWN!')
    global vectorstore
    vectorstore.persist()
    vectorstore = None
    sys.exit(0)
