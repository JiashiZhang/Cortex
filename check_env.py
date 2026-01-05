import sys
import importlib.metadata

packages = [
    "langchain",
    "langchain-community",
    "langchain-core",
    "langchain-google-genai",
    "numpy"
]

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 40)

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: Not Installed")

print("-" * 40)
try:
    import langchain.agents
    print(f"langchain.agents location: {langchain.agents.__file__}")
    has_agent = hasattr(langchain.agents, 'create_tool_calling_agent')
    print(f"Has create_tool_calling_agent: {has_agent}")
    if not has_agent:
        print(f"Available in agents: {dir(langchain.agents)}")
except Exception as e:
    print(f"Error importing langchain.agents: {e}")
