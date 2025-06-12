from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Initialize app
app = Flask(__name__)

# Initialize global variables
db = None
qa = None

def initialize_ai():
    global db, qa
    try:
        # Load and process your documents with smaller chunks
        loader = TextLoader("Texts/Ulysses.txt")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        
        # Embed and store in FAISS vector DB
        embedding = OpenAIEmbeddings()
        db = FAISS.from_documents(split_docs, embedding)
        
        # Set up retriever and chain
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4o"), retriever=retriever)
        print("AI initialization successful!")
        return True
    except Exception as e:
        print(f"AI initialization failed: {e}")
        return False

# Flask route for Thunkable to send a message
@app.route('/ask', methods=['POST'])
def ask():
    global qa
    if qa is None:
        return jsonify({"reply": "AI system not initialized. Please check your OpenAI API key and quota."})
    
    try:
        data = request.get_json()
        question = data.get("question")
        response = qa.run(question)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"reply": f"Error processing question: {str(e)}"})

# Test route to check if server is running
@app.route('/')
def home():
    return "Flask server is running! Use POST /ask to query the AI."

# Start Flask app
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Attempting to initialize AI system...")
    initialize_ai()  # Try to initialize, but don't fail if it doesn't work
    print("Flask server starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
