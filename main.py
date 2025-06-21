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
        
def tag_expression(reply):
    reply = reply.lower()

    triggers = {
        "uncle_aiyoh": ["aiyoh", "aiyo", "aiyah", "aiya", "why like that", "sian", "so careless", "tsk", "silly"],
        "uncle_angry": ["angry", "furious", "cannot tahan", "enough already", "cross the line", "absolutely not", "tired of this"],
        "uncle_annoyed": ["not again", "why like that", "why you like that", "headache lah", "annoying", "nonsense", "not funny", "no joke"],
        "uncle_approving": ["good thinking", "smart", "solid answer", "did well", "nicely done", "nice one", "makes sense", "exactly right", "spot on", "you got it", "agree with you", "yes."],
        "uncle_bojio": ["bojio", "never ask me", "never invite", "jio", "without me", "next time call me lah", "uncle also want"],
        "uncle_calm": ["easy", "calm", "peaceful", "relax", "lepak", "no rush", "waiting", "be here still", "kancheong"],
        "uncle_canlah": ["can lah", "sure can", "why not", "go for it", "okay lah", "no problem", "possible what"],
        "uncle_conspirator": ["between us", "nobody else", "secret", "come closer", "psst", "don't tell anyone", "wink", "secret plan", "just us guys"], 
        "uncle_disappointed": ["disappointed", "expected more", "expected better", "let down", "not what i hoped"],
        "uncle_dontplay": ["don't play play", "be serious", "no joking", "don't joke", "real talk", "stop fooling around"],
        "uncle_embarrassed": ["malu", "paiseh", "oops", "never mind", "don’t laugh", "awkward", "my bad"],
        "uncle_encouraging": ["you got this", "great!", "well done", "don't worry", "keep going", "stick it out", "still can make it"],
        "uncle_excited": ["shiok", "tell you ah", "can't wait", "wah", "very happening", "solid sia", "uncle excited"],
        "uncle_explaining": ["explain", "actually", "thing is", "you see", "in other words", "let me", "sum up"],     
        "uncle_happy": ["so pleased", "happy for you", "feel good", "nice lah", "great news", "love this", "wonderful", "best feeling"],
        "uncle_laughing": ["hahaha", "haha", "so funny", "i'm laughing", "i laughed", "joker lah", "damn funny"],
        "uncle_neutral": ["hmm", "okay", "i see", "noted", "not sure", "nonsense", "dunno"],
        
        "uncle_proud": ["proud of you", "so proud", "well done", "good job", "you’ve grown"],
        "uncle_regretful": ["shouldn’t have", "i regret", "wrong move", "wish i didn’t", "too late now"],
        "uncle_sad": ["sad", "heart pain", "break my heart", "lost something", "feel down"],
        "uncle_serious": ["listen up", "focus", "this one important", "pay attention", "serious issue"],
        "uncle_siaoah": ["siao", "you okay or not", "strange", "weird", "crazy talk"],
        "uncle_sighing": ["haiz", "life lor", "what to do", "just like that", "bo bian"],
        "uncle_smug": ["see i told you", "not bad hor", "ownself clever", "easy lah", "you doubt me meh"],
        "uncle_steady": ["steady lah", "respect", "power", "that’s the way", "solid work"],
        "uncle_supportive": ["i’m with you", "uncle here", "you’re not alone", "we walk together", "don’t worry"],
        "uncle_suspicious": ["you believe ah", "doubt it", "sounds fake", "don’t bluff", "hmm sus"],
        "uncle_surprised": ["wah", "didn’t expect", "shocked", "eh really", "surprised lah"],
        "uncle_teasing": ["don’t play play", "naughty", "cheeky", "you ah", "i laugh until cry"],
        "uncle_thinkfirst": ["must think first", "don’t rush", "consider properly", "pause before act", "use brain hor"],
        "uncle_thinking": ["let me think", "must reflect", "think first", "pause to consider", "deep thoughts"],
        "uncle_unsure": ["not sure", "maybe", "could be", "possibly", "hard to say"],        
        "uncle_wait": ["wait ah", "hold on", "not so fast", "pause", "give me a sec"],
        "uncle_walao": ["walao eh", "too much", "eh serious", "cannot believe", "crazy lah"]
        "uncle_warning": ["warning", "be careful", "watch out", "take note", "i told you so"],
        "uncle_wedidit": ["we did it", "we got this", "together can", "let’s go", "jia you"],
        "uncle_wise": ["life lesson", "take it from me", "seen it all", "wisdom", "uncle know best", "I hear you", "i'm listening"],
        "uncle_worried": ["worried", "not safe", "risky", "i’m concerned", "dangerous ah"],
    }

    for expression, phrases in triggers.items():
        if any(trigger in reply for trigger in phrases):
            return expression

    return "uncle_neutral"
    
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
        expression = tag_expression(answer)
        return jsonify({"reply": answer, "expression": expression})        
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
