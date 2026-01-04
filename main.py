import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Import tools
from tools.memory_store import save_memory, search_memory

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
    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.success("OpenAI API Key loaded from environment.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Agent (Lazy initialization to handle API Key)
if "agent_executor" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_memory, search_memory]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful personal assistant acting as a 'Second Brain'. "
                   "You have access to a long-term memory storage. "
                   "When the user tells you something they want to remember, use the 'save_memory' tool. "
                   "When the user asks about something they might have told you before, use the 'search_memory' tool. "
                   "Always answer in the language the user speaks (mostly Chinese or English)."),
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
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide an OpenAI API Key.")
        else:
            try:
                # Prepare chat history for the agent
                # We only pass the last few messages to keep context window manageable if needed,
                # but for now, let's pass the conversation history formatted for LangChain if we were using memory in the agent.
                # However, create_tool_calling_agent usually expects 'chat_history' as a list of messages.
                # Since we are using a simple stateless run here (agent executor doesn't persist state automatically across streamlit reruns unless we manage it),
                # we will just pass the current input. For a better chat experience, we should construct chat_history.
                
                # Simple approach: Just pass the input. The agent has tools to look up long-term memory.
                # Context from the immediate conversation is less critical for this specific "Second Brain" demo 
                # unless we implement short-term memory handling.
                
                response = st.session_state.agent_executor.invoke({"input": prompt, "chat_history": []})
                output_text = response["output"]
                
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
