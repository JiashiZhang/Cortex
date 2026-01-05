import os
import warnings
import logging

# Filter out specific warnings to keep the console clean
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Suppress specific logging from google libraries if needed
logging.getLogger("google.generativeai").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Import tools
from tools.memory_store import save_memory, search_memory, extract_and_save_knowledge

# Page Config
st.set_page_config(page_title="Private Second Brain", page_icon="🧠")

st.title("🧠 Private Second Brain")
st.markdown("""
> A local AI Agent acting as your second brain. 
> It remembers your thoughts and retrieves them when needed.
""")

# Sidebar for API Key configuration (optional if .env is used)
with st.sidebar:
    st.header("Settings")
    if not os.getenv("GOOGLE_API_KEY"):
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("Google API Key loaded from environment.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Agent (Lazy initialization to handle API Key)
if "agent_executor" not in st.session_state:
    # Use Gemini model
    # Get model name from environment variable, default to gemini-flash-latest if not set
    model_name = os.getenv("GOOGLE_MODEL", "gemini-flash-latest")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    # Updated tool list with knowledge graph capabilities
    tools = [save_memory, search_memory, extract_and_save_knowledge]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful personal assistant acting as a 'Second Brain'. "
                   "You have access to a long-term memory storage with Knowledge Graph capabilities. "
                   "\n\n"
                   "CORE INSTRUCTIONS:\n"
                   "1. **Save Memories**: When the user tells you something they want to remember, use 'save_memory'. "
                   "If the information contains a clear relationship between two concepts (e.g., 'Alice is Bob's manager'), "
                   "PREFER using 'extract_and_save_knowledge' to build the knowledge graph.\n"
                   "2. **Retrieve Information**: When the user asks about something, use 'search_memory'. "
                   "This tool now automatically checks both semantic notes and the knowledge graph.\n"
                   "3. **Proactive Context**: If the user mentions a topic you have stored knowledge about, "
                   "you can subtly mention related connections you found in the graph to show 'awareness'.\n"
                   "4. **Language**: Always answer in the language the user speaks (mostly Chinese or English)."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Chat Input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide a Google API Key.")
        else:
            try:
                # Prepare chat history for the agent
                response = st.session_state.agent_executor.invoke({"input": prompt, "chat_history": []})
                output_text = response["output"]
                
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                # Filter out the specific Pydantic warning if it appears in the error message
                error_msg = str(e)
                if "Key 'title' is not supported in schema" in error_msg:
                    # This is a warning, not a fatal error, but we can log it or ignore it
                    pass 
                else:
                    st.error(f"An error occurred: {e}")
