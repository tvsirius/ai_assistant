# ai_assistant
This is a test AI assistant with memory to talk to chatGPT


Task:

AI Assiatant with memory
Using LangChain library and OpenAI API, create an application that allows user to speak to GPT-3.5-turbo. UI interface does not matter. 
Application should save previous conversation history into a memory (Chroma), and when new conversion starts - retrieve previous conversations from memory and use them in the prompt to set the context.


Working ver with TODO  
1. Chromadb - understanding 
2. Moving from file to db or chromastorage (?)
3. Adding ConversationSummaryMemory and combined prompt, to fit into 4096 tokens and provide better context.
4. Voice in and out conversation
