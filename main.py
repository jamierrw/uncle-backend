from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize app
app = Flask(__name__)

# Load and process your documents
loader = TextLoader("Texts/Ulysses.txt")  # Start with one file
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embed and store in FAISS vector DB
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embedding)

# Set up retriever and chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4o"), retriever=retriever)

# Flask route for Thunkable to send a message
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")
    response = qa.run(question)
    return jsonify({"reply": response})

# Start Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)
