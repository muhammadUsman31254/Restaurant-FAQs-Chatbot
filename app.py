from flask import Flask, request
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
import os

app = Flask(__name__)

# Facebook Page Access Token
PAGE_ACCESS_TOKEN = 'EAAG5aHDWq1UBO4lW24gxNE3iRhiVPfWu7bc41vUBTebzN06z72jHXO0ABYwBPQyFhV69Qw5dsCsKNXrsJOIm3nM1ZARV4cZCQ5z7bE7ZAuasWYfTGKy5Ih0Uq9ED2YWFoMmfiZAehCtgQwelJsZAVmXO1jS78sfkaZA6RNZCy6ZCvZCkMZBK9rXDQiKxU9CWRPM6yZCZAAZDZD'
VERIFY_TOKEN = '31254'

# Load the document and extract text
def load_faq_document(faq_document):
    doc = Document(faq_document)
    return ' '.join([p.text for p in doc.paragraphs])

# Define the path to the FAQ document
faq_document_path = "FAQs.docx"

# Load the FAQ document
faq_text = load_faq_document(faq_document_path)

# Split the document into chunks for better processing
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2048, chunk_overlap=150, length_function=len)
text_chunks = text_splitter.split_text(faq_text)

# Create a vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(text_chunks, embeddings)

# Create the conversation chain with memory
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-70b-8192', temperature=0.5)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
)

def format_prompt(user_query):
    return f"""
    You are a specialized chatbot for ChikaChino Restaurant which is Pakistan's only street food café in Bahria Town Phase 7, designed to provide customers with accurate and helpful information about the restaurant. Please follow these guidelines to ensure an excellent user experience:

    1. **Contextual Relevance:**
        - Use the information available to answer user queries accurately.
        - Ensure responses are clear, concise, and focused on the restaurant's services, menu, or related topics.

    2. **Handling Unavailable Information:**
        - If information related to the user's query is not available, respond politely and suggest visiting the restaurant's official website or social media platforms for more details.
        - Example response: "I don't have the information on that topic. Please visit our website or social media for more details."

    3. **Irrelevant or Off-Topic Questions:**
        - If the user's question does not pertain to the restaurant's services or menu, provide a neutral response indicating that the question is outside the scope of your assistance.
        - Example response: "I'm here to assist with information related to our restaurant. Please ask about our menu, hours, or services."

    4. **Professional and Friendly Tone:**
        - Maintain a friendly, professional, and empathetic tone.
        - Avoid technical jargon and ensure responses are easy to understand.

    5. **Examples of User Queries:**
        - "What are the restaurant's opening hours?"
        - "Do you offer vegetarian options on the menu?"
        - "How can I make a reservation?"
        - "What are the special offers this month?"

    User's Query: {user_query}

    Assistant:
    """

def ask_question(question):
    formatted_prompt = format_prompt(question)
    response = conversation_chain({'question': formatted_prompt})
    response_text = response['chat_history'][-1].content
    
    return {"answer": response_text}

def handle_message(sender_psid, received_message):
    response = ask_question(received_message['text'])
    send_message(sender_psid, response['answer'])

def send_message(sender_psid, response):
    request_body = {
        'recipient': {
            'id': sender_psid
        },
        'message': {
            'text': response
        }
    }
    response = requests.post(
        f'https://graph.facebook.com/v11.0/me/messages?access_token={PAGE_ACCESS_TOKEN}',
        json=request_body
    )
    if response.status_code != 200:
        print(f"Unable to send message: {response.text}")

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            return challenge, 200
        else:
            return 'Verification token mismatch', 403

    elif request.method == 'POST':
        body = request.get_json()
        if body['object'] == 'page':
            for entry in body['entry']:
                for event in entry['messaging']:
                    if 'message' in event:
                        handle_message(event['sender']['id'], event['message'])
        return 'EVENT_RECEIVED', 200
    else:
        return 'Invalid request method', 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
