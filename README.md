# ai_assistant
This is a test AI assistant with memory to speak with chatGPT


Task:

AI Assiatant with memory
Using LangChain library and OpenAI API, create an application that allows user to speak to GPT-3.5-turbo. UI interface does not matter. 
Application should save previous conversation history into a memory (Chroma), and when new conversion starts - retrieve previous conversations from memory and use them in the prompt to set the context.




current version 1.0 beta


All task goals accomplished.

Some testing is needed, but generally all seems working. 

Speech-to-text and text-to-speech gives a conversation with AI Assistant generally new feelings (text-to-speech for now works with English only)


with ai_roles.py and AI_ASSISTANT_ROLE= var in server.py is a way to switch between a predefined set of assistant roles


TODO in the future:
1. Add a langchain agent to process general comands like clear history, and switch Assistant roles
2. Find more pleasant voice for text-to-speech, and enable it to speak different languages with autodetect
3. Think about implementation of continues voice conversation - finding a way to automatically turn voice input record on and off, so you can speak like with a real person
