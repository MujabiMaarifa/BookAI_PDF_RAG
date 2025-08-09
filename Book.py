from langchain import hub
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
import os
from dotenv import load_dotenv

#models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

#define embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#define the system messages
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


load_dotenv()

os.environ["USER_AGENT"] = "MyNLPGeeksforgeeksRAG/1.0 (daudimujabi@gmail.com)"
#api keys
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not google_api_key:
   print("The api key for google germini is not found")
   exit()

if not openai_api_key:
    print("The api key for openai is not found")
    exit()


#load the llm models
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
openai_llm = init_chat_model(model="gpt-4o-mini", openai_api_key=openai_api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

#select and define the vector store
vector_store = InMemoryVectorStore(embeddings)

#define the document loader
loader = PyMuPDFLoader("programming.pdf")

#load the document using the loader extension
docs = loader.load()

#split the texts using the recursive text splitter character for easy and clear access of content of the defined document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000, chunk_overlap = 300
)
add_splits = text_splitter.split_documents(docs)

#index chunks
_= vector_store.add_documents(
    documents = add_splits
)

#define the prompt to be used by the user

prompt = hub.pull("rlm/rag-prompt")

# define the state for the application
class State(TypedDict) :
    question: str
    context: List[Document]
    answer: str
    chat_history: List[BaseMessage]

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=8)
    return {"context": retrieved_docs}

def generate(state: State):
    messages =[]

    #add a message history
    if state["chat_history"]:
        messages.extend(state["chat_history"])
    
    messages.append(HumanMessage(content=user_input))
    if state['context']:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        user_prompt = prompt.invoke({
            "question":state['question'],
            "context":docs_content
        })
        messages.extend(user_prompt.to_messages())
        response = gemini_llm.invoke(messages)
        
    #fall back generation
    else:
        user_prompt = prompt.invoke({
            "question":f"You are Mujabi Maarifa Bot. Answer the question using your own knowledge {state['question']}",
            "context":""
            })
        messages.extend(user_prompt.to_messages())
        response = openai_llm.invoke(messages)
        

    messages.append(AIMessage(content=response.content))

    return {
        "answer": response.content,
        "chat_history": messages
    }

#compile app application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("\n✨Welcome to Maarifa Programming Language Bot✨")
print("Type your question below. Type 'exit' or 'quit' to end the conversation.\n")

if __name__ == "__main__":
    chat_history = []

    while True:
        user_input = input("You🔍: ").strip()
        greetings = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "bonjour"}

        if user_input.lower() in greetings:
            print("Assistant🧠: Hello! How can I help you with programming notes today?\n")
            continue
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant🧠: Goodbye! Feel free to reach out for any queries...")
            break

        response = graph.invoke({
            "question": user_input,
            "chat_history": chat_history
            
            })
        print(f"Assistant🧠: {response['answer']}\n")

        chat_history = response["chat_history"]