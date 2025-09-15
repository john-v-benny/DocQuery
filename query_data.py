import argparse
import os
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "D:/rag/chroma_db"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class LocalEmbeddingFunction(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load Chroma DB with local embedding function
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_function = LocalEmbeddingFunction(model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use Gemini as LLM
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    response_text = response.text

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()