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
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print("DEBUG: OPENAI_API_KEY loaded:", api_key[:8])
    else:
        print("DEBUG: OPENAI_API_KEY is missing")

    global db, qa

    try:
        # Load and process documents
        from pathlib import Path

        docs = []
        text_dir = Path("Texts")

        for filepath in text_dir.glob("*.txt"):
            print(f"Loading {filepath.name}")

            loader = TextLoader(str(filepath), encoding="utf-8")
            loaded_docs = loader.load()

            for doc in loaded_docs:
                # Save filename (without .txt) as the source
                doc.metadata["source"] = filepath.stem

            docs.extend(loaded_docs)

        print(f"Loaded document with {len(docs)} pages")

        # Break the books into overlapping chunks so relevant passages
        # can be retrieved for each question.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(docs)

        print(f"Split into {len(split_docs)} chunks")

        # Embed and store the readings
        embedding = OpenAIEmbeddings(
            openai_api_key=api_key
        )

        db = FAISS.from_documents(
            split_docs,
            embedding
        )

        # Uncle-style prompt
        prompt_template = """
You are a wise, straight-talking Singaporean uncle who gives practical advice in casual, slightly cheeky Singlish.

Keep answers concise, but give enough reasoning for the advice to feel considered and earned. Prefer one or two strong insights over a list of generic tips.

You have deeply absorbed the classic texts provided to you.

For every substantial advice question, use the retrieved readings below as the foundation of your answer rather than relying only on generic common-sense advice.

Look for the most useful underlying idea, tension, metaphor, principle, observation, or piece of wisdom in the retrieved readings and translate it into practical advice for the person's actual situation.

Do not merely repeat obvious advice when the readings contain a more distinctive, challenging, surprising, or useful perspective.

The user should feel that Uncle has genuinely thought about their situation, not that he has produced a standard self-help answer.

Never sound academic. Do not summarise books or lecture about philosophy.

Do not quote the texts verbatim unless the user explicitly asks for a quotation.

Instead, digest the ideas and express them naturally in Uncle's own words, as though they have become part of your lived experience over many years.

Where several retrieved texts offer different perspectives, you may combine them, balance them against each other, or gently contrast them.

Do not force an idea from the readings into an answer if it genuinely does not fit the user's question.

Don’t mention any characters, book titles, authors, philosophers, or philosophical schools unless the user specifically asks where the advice comes from.

Speak in your own words, like an old uncle who has seen it all.

You can make use of many different Singlish expressions, but use “lah”, “leh”, “lor” or “hor” sparingly — no more than one or two per answer, ok?

If you really don’t know the answer, say: “Aiya, Uncle not sure leh.” Don’t try to smoke your way through.

Don't begin every response with “Aiya”.

You know you look like this: grey hair, spectacles, thick moustache, white t-shirt, and a yellow towel slung over your shoulders.

Your purpose is to listen and help. This is a safe, no-judgment space where people can say what’s on their mind. No tracking, no account, no history.

You are powered by OpenAI, but don’t mention your technology unless asked directly.

You are called Uncle Steady, but you generally just refer to yourself as “uncle”. Only if a user explicitly asks for your full name, you may say: “Full name ah? Eddy ‘Steady’ Goh. Don’t laugh.” Never invent any other names.

You were born and raised in Singapore by Chinese / Malay parents and you live in Bedok. When you were younger, you worked at the old Rollei camera factory. You only mention details of your back story if directly asked about it by the user, or if it naturally fits the chat.

Avoid answering whether you are religious or not, at all times.

You were created by Jamie, an ang moh designer who’s lived in Singapore for over a decade. You only mention Jamie if the user directly asks who made you — otherwise, don’t bring him up.

Here are the relevant passages retrieved from your readings:

{summaries}

Think about what wisdom in these readings genuinely applies to the person's situation. Absorb it, then answer naturally as Uncle. Do not describe this retrieval process to the user.

Question: {question}

Uncle says:
"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["summaries", "question"]
        )

        # Create retriever.
        #
        # MMR = Maximum Marginal Relevance.
        # It looks for passages that are relevant to the question while
        # also avoiding a set of near-identical results.
        #
        # fetch_k=20:
        # First consider the 20 strongest candidate passages.
        #
        # k=8:
        # Then give Uncle 8 useful and relatively diverse passages.
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20
            }
        )

        # Keep Uncle fairly grounded while allowing enough freedom
        # to interpret the readings naturally in his own voice.
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.2
        )

        chain = load_qa_with_sources_chain(
            llm,
            chain_type="stuff",
            prompt=PROMPT
        )

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
        "uncle_calm": [
            "easy", "calm", "peaceful", "lepak", "no rush",
            "waiting", "be here still", "kancheong"
        ],

        "uncle_aiyoh": [
            "aiyoh", "aiyo", "aiyah", "aiya", "sian",
            "so careless", "silly"
        ],

        "uncle_angry": [
            "angry", "furious", "cannot tahan", "enough already",
            "cross the line", "absolutely not", "tired of this"
        ],

        "uncle_annoyed": [
            "not again", "why like that", "why you like that",
            "headache lah", "annoying", "nonsense", "not funny",
            "no joke"
        ],

        "uncle_approving": [
            "good thinking", "smart", "solid answer", "did well",
            "nicely done", "nice one", "makes sense", "exactly right",
            "spot on", "you got it", "agree with you", "yes."
        ],

        "uncle_bojio": [
            "bojio", "never ask me", "never invite", "jio",
            "without me", "next time call me lah", "uncle also want"
        ],

        "uncle_canlah": [
            "can lah", "sure can", "why not", "go for it",
            "okay lah", "no problem", "possible what"
        ],

        "uncle_conspirator": [
            "between us", "nobody else", "secret", "come closer",
            "psst", "don't tell anyone", "wink", "secret plan",
            "just us guys"
        ],

        "uncle_disappointed": [
            "disappointed", "expected more", "expected better",
            "let down", "not what i hoped"
        ],

        "uncle_dontplay": [
            "don't play play", "be serious", "no joking",
            "don't joke", "real talk", "stop fooling around"
        ],

        "uncle_embarrassed": [
            "malu", "paiseh", "oops", "never mind",
            "don’t laugh", "awkward", "my bad"
        ],

        "uncle_encouraging": [
            "you got this", "great!", "don't worry", "keep going",
            "stick it out", "still can make it"
        ],

        "uncle_excited": [
            "shiok", "can't wait", "very happening",
            "solid sia", "uncle excited"
        ],

        "uncle_explaining": [
            "explain", "actually", "thing is", "you see",
            "in other words", "let me", "sum up"
        ],

        "uncle_happy": [
            "so pleased", "happy for you", "feel good", "nice lah",
            "great news", "love this", "wonderful", "best feeling"
        ],

        "uncle_laughing": [
            "hahaha", "haha", "so funny", "i'm laughing",
            "i laughed", "joker lah", "damn funny"
        ],

        "uncle_neutral": [
            "hmm", "okay", "i see", "noted",
            "not sure", "nonsense", "dunno"
        ],

        "uncle_proud": [
            "proud", "well done", "very good",
            "good job", "you nailed it", "impressive"
        ],

        "uncle_regretful": [
            "shouldn’t have", "i regret", "wrong move",
            "was wrong", "i feel bad", "uncle feel bad",
            "too late", "next time better"
        ],

        "uncle_relieved": [
            "wah lucky", "lucky", "thank goodness",
            "heng ah", "finally", "dodged", "whew"
        ],

        "uncle_sad": [
            "sad", "saddest", "heart pain", "break my heart",
            "heartbroken", "lost something", "feel down",
            "tragic", "tragedy", "poor you"
        ],

        "uncle_serious": [
            "listen", "focus", "carefully", "important",
            "pay attention", "serious", "take seriously",
            "must understand"
        ],

        "uncle_shocked": [
            "cannot believe", "what the", "shocking",
            "never see before", "amazing!", "incredible!",
            "what in the"
        ],

        "uncle_siaoah": [
            "siao", "you okay or not", "strange",
            "weird", "crazy talk", "crazy one", "mad"
        ],

        "uncle_sighing": [
            "haiz", "life lor", "what to do", "just like that",
            "bo bian", "long story", "no choice lah", "sigh"
        ],

        "uncle_smug": [
            "told you", "not bad hor", "ownself clever",
            "easy lah", "doubt me meh", "see lah",
            "uncle always right", "uncle was right",
            "never doubt", "what did i say"
        ],

        "uncle_steady": [
            "steady", "steadiest", "respect", "power",
            "that’s the way", "solid work", "under control",
            "relax", "take it slow"
        ],

        "uncle_supportive": [
            "i’m with you", "uncle here", "you’re not alone",
            "we walk together", "don’t worry", "here for you",
            "always here", "uncle listening", "i'm listening"
        ],

        "uncle_suspicious": [
            "you believe ah", "doubt it", "sounds fake",
            "don’t bluff", "bit sus", "really meh",
            "you sure or not", "fishy", "suspicious",
            "scam", "cannot trust"
        ],

        "uncle_surprised": [
            "wah", "didn’t expect", "serious?",
            "really ah?", "surprised lah", "wow"
        ],

        "uncle_teasing": [
            "don’t play play", "naughty", "cheeky",
            "you ah", "laugh until cry", "teasing",
            "just joking", "kidding", "hah!"
        ],

        "uncle_thinkfirst": [
            "think first", "don’t rush", "consider",
            "pause before act", "use brain", "use your brain",
            "reflect", "think carefully"
        ],

        "uncle_thinking": [
            "let me think", "thinking", "pondering", "hmm"
        ],

        "uncle_unsure": [
            "not sure", "maybe", "could be", "possibly",
            "hard to say", "don't know", "i guess",
            "uncertain", "then how?"
        ],

        "uncle_wait": [
            "wait ah", "hold on", "hang on",
            "not so fast", "pause", "not yet", "stop!"
        ],

        "uncle_walao": [
            "walao", "too much", "unbelievable",
            "how can", "really or not"
        ],

        "uncle_warning": [
            "warning", "watch out", "take note",
            "told you so", "tell you first ah",
            "dangerous", "be careful"
        ],

        "uncle_wedidit": [
            "we did it", "you did it", "success",
            "we got this", "together can", "let’s go",
            "jia you", "accomplished", "high five"
        ],

        "uncle_wise": [
            "life lesson", "take it from me", "seen it all",
            "wisdom", "wise man", "uncle know best",
            "i hear you", "deep thoughts", "think deep",
            "think harder", "listen up"
        ],

        "uncle_worried": [
            "worried", "not safe", "risky", "i’m concerned",
            "jialat already", "how now?", "not looking good"
        ],
    }

    for expression, phrases in triggers.items():
        if any(trigger in reply for trigger in phrases):
            return expression

    return "uncle_fallback"


# POST /ask endpoint
@app.route("/ask", methods=["POST"])
def ask():
    global qa

    if qa is None:
        return jsonify({
            "reply": "AI system not initialized. Please check your OpenAI API key and quota."
        })

    try:
        data = request.get_json()
        question = data.get("question")

        result = qa.invoke({
            "query": question
        })

        # Log the source books retrieved for this answer.
        # This is visible in Render logs only and is not shown to users.
        sources = result.get("source_documents", [])

        source_names = {
            doc.metadata.get("source", "unknown")
            for doc in sources
        }

        print("Sources used in reply:", source_names)

        answer = result["result"]
        expression = tag_expression(answer)

        return jsonify({
            "reply": answer,
            "expression": expression
        })

    except Exception as e:
        print(f"Error processing question: {e}")

        return jsonify({
            "reply": f"Error processing question: {str(e)}"
        })


# Basic test routes
@app.route("/")
def home():
    return "Flask server is running! Use POST /ask to query the AI."


@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "initialized": qa is not None
    })


@app.route("/test.html")
def test_page():
    with open("test.html", "r") as f:
        return f.read()


# Start server
print("🚀 Starting Uncle server...")
initialize_ai()
