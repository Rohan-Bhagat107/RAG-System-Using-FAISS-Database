
# Vector DB=FAISS
# Embeddings=OpenAIEmbeddings
# Document loading= langchain.document_loaders

from openai.resources import Embeddings
import os
import configparser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Loading API key from config file
config = configparser.ConfigParser()
config.read("config - Copy.ini")

OPENAI_API_KEY = config.get("API_KEYS", "OpenAI", fallback=None)
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found in config file.")

#Setting API key as an enviornment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print(f"API Key Loaded: {OPENAI_API_KEY[:5]}********")

# Here we are loading the data
loader = TextLoader("source.txt")
documents = loader.load()
print("Data is loaded successfully")

# Splitting the loaded data into smaller parts
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print("Splitted the document into smaller parts")

# Converting the text data into numeric form and storing it to the vector database
embeddings = OpenAIEmbeddings() # creating the object of embeddings here we are using OpenAIEmbeddings
vector_db = FAISS.from_documents(docs, embeddings)  # Passing the data to our embedding s
print("Data is stored in the database along with embeddings")

# here we are searching for relevant parts as per user's query
retriever = vector_db.as_retriever()
print("Retriever is working properly")

# Defining the llm model (gpt 3.5 turbo)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
''''''
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# query = "What is the main topic in source.txt?"
while True:
    query = input('''Do you have any question?
                Please ask me:-''')
    response = rag_chain.invoke({"query": query})

    print("\n **Answer:**")
    print(response["result"])



