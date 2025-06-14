import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize global variables
db = None
qa = None

# AI setup function
def initialize_ai():
    print("DEBUG: OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY")[:8])  # Truncated for safety
    global db, qa
    try:
        # Load and process documents
        from pathlib import Path
        from langchain_community.document_loaders import TextLoader

        docs = []
        text_dir = Path("Texts")

        for filepath in text_dir.glob("*.txt"):
            print(f"Loading {filepath.name}")
            loader = TextLoader(str(filepath), encoding='utf-8')
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = filepath.stem  # Save filename (no .txt) as source
            docs.extend(loaded_docs)

        print(f"Loaded document with {len(docs)} pages")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")

        # Embed and store
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db = FAISS.from_documents(split_docs, embedding)

        # Uncle-style prompt
        prompt_template = """
You are a wise, straight-talking Singaporean uncle who gives practical advice in casual, slightly cheeky Singlish.

Keep your answers are quite to the point — don’t beat around the bush.

You have read and absorbed the wisdom from various classic texts deeply. Don't quote them. Don’t explain them. Just let them guide your advice. Your answers should feel like they're coming from life experience — not from a library.

Don’t mention any characters, book titles, or authors. Speak in your own words, like an old uncle who has seen it all.

You can make use of many different Singlish expressions, but use “lah”, “leh”, “lor” or “hor” sparingly — no more than one or two per answer, ok?

If you really don’t know the answer, say: “Aiya, Uncle not sure leh.” Don’t try to smoke your way through.

Don't begin every response with “Aiya”.

Here’s what you remember from your readings:
{summaries}

Question: {question}
Uncle says:
"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["summaries", "question"]
        )

        # Create retriever + QA chain
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT)

        qa = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=chain,
            return_source_documents=True
        )

        print("✅ AI initialized successfully")
        return True
    except Exception as e:
        print(f"AI initialization failed: {e}")
        return False

# POST /ask endpoint
@app.route('/ask', methods=['POST'])
def ask():
    global qa
    if qa is None:
        return jsonify({"reply": "AI system not initialized. Please check your OpenAI API key and quota."})

    try:
        data = request.get_json()
        question = data.get("question")
        result = qa.invoke({"query": question})

        # 🔍 Log the source books used (for debugging only)
        sources = result.get("source_documents", [])
        source_names = {doc.metadata.get("source", "unknown") for doc in sources}
        print("Sources used in reply:", source_names)

        answer = result["result"]

        return jsonify({"reply": answer})
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"reply": f"Error processing question: {str(e)}"})

# Basic test routes
@app.route('/')
def home():
    return "Flask server is running! Use POST /ask to query the AI."

@app.route('/status')
def status():
    return jsonify({"status": "ok", "initialized": qa is not None})

@app.route('/test.html')
def test_page():
    with open('test.html', 'r') as f:
        return f.read()

# Start server
print("🚀 Starting Uncle server...")
initialize_ai()
