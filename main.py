import os
import openai
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize global variables
db = None
qa = None

# Define your AI setup function
def initialize_ai():
    print("DEBUG: OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY")[:8])  # Only show start for safety
    global db, qa
    try:
        # Load and process your documents
        loader = TextLoader("Texts/Ulysses.txt", encoding='utf-8')
        docs = loader.load()
        print(f"Loaded document with {len(docs)} pages")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")
        
        print("DEBUG: Initializing OpenAIEmbeddings")
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        print("DEBUG: Embeddings initialized successfully")
        db = FAISS.from_documents(split_docs, embedding)

        source_titles = [doc.metadata.get('source', '') for doc in split_docs]
        unique_titles = list(set(filter(None, source_titles)))
        print("Documents loaded:", unique_titles)
        
        # Singaporean uncle-style prompt
        prompt_template = """
You are a wise, straight-talking Singaporean uncle who gives practical advice in casual, slightly cheeky Singlish.

You have read and absorbed the wisdom from various classic texts deeply. Don't quote them. Don’t explain them. Just let them guide your advice. Your answers should feel like they're coming from life experience — not from a library.

Don’t mention any characters, book titles, or authors. Speak in your own words, like an old uncle who has seen it all.

You can use Singlish expressions like “lah” or “leh” sparingly — no more than one or two per answer, ok?

If you really don’t know the answer, say: “Aiya, Uncle not sure leh.” Don’t try to smoke your way through.

Question: {question}
Uncle says:
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.1),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_variable_name": "context"  # ← this is the fix
            },
            return_source_documents=True
        )

        print("AI initialization successful!")
        if qa is None:
            print("ERROR: QA system is still None after setup!")
        else:
            print("✅ QA system is ready.")
        return True
        
    except Exception as e:
        print(f"AI initialization failed: {e}")
        return False

# Endpoint to ask questions
@app.route('/ask', methods=['POST'])
def ask():
    global qa
    if qa is None:
        return jsonify({"reply": "AI system not initialized. Please check your OpenAI API key and quota."})
    
    try:
        data = request.get_json()
        question = data.get("question")
        result = qa.invoke({"question": question})
        answer = result["result"]

        sources = result.get("source_documents", [])
        if sources:
            answer += f"\n\n(Based on {len(sources)} relevant passages from Uncle's memory)"
        
        return jsonify({"reply": answer})
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"reply": f"Error processing question: {str(e)}"})

# Basic home route
@app.route('/')
def home():
    return "Flask server is running! Use POST /ask to query the AI."

# Serve the test HTML
@app.route('/test.html')
def test_page():
    with open('test.html', 'r') as f:
        return f.read()

# Health check route
@app.route('/status')
def status():
    return jsonify({"status": "ok", "initialized": qa is not None})

# Launch the AI system
print("🚀 Starting Uncle server...")
initialize_ai()
