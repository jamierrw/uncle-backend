from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
        # Load and process your documents with better text splitting
        loader = TextLoader("Texts/Ulysses.txt", encoding='utf-8')
        docs = loader.load()
        print(f"Loaded document with {len(docs)} pages")
        
        # Use RecursiveCharacterTextSplitter for better chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")
        
        # Embed and store in FAISS vector DB
        embedding = OpenAIEmbeddings()
        db = FAISS.from_documents(split_docs, embedding)
        
        # Create a custom prompt template
        prompt_template = """Use the following pieces of context from James Joyce's Ulysses to answer the question. If you don't know the answer based on the context, just say that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Set up retriever and chain with better configuration
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.1),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
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
        
        # Use invoke instead of deprecated run method
        result = qa.invoke({"query": question})
        
        # Extract the answer from the result
        answer = result["result"]
        
        # Optionally include source information
        sources = result.get("source_documents", [])
        if sources:
            answer += f"\n\n(Based on {len(sources)} relevant passages from Ulysses)"
        
        return jsonify({"reply": answer})
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"reply": f"Error processing question: {str(e)}"})

# Test route to check if server is running
@app.route('/')
def home():
    return "Flask server is running! Use POST /ask to query the AI."

# Route to serve the test page
@app.route('/test.html')
def test_page():
    with open('test.html', 'r') as f:
        return f.read()

# Start Flask app
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Attempting to initialize AI system...")
    initialize_ai()  # Try to initialize, but don't fail if it doesn't work
    print("Flask server starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
