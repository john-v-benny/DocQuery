from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

load_dotenv()


DATA_PATH = "D:/rag/data/xylos.pdf"
CHROMA_PATH = "D:/rag/chroma_db"

def load_data():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks of text")

    return chunks

class LocalEmbeddingFunction(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_function = LocalEmbeddingFunction(model)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print(f"Saved {len(chunks)} to ChromaDB at {CHROMA_PATH}")


def main():
    document = load_data()
    print(f"Loaded {len(document)} pages!")
    chunks = split_text(document)
    #print(f"First chunk sample:\n{chunks[0].page_content}")
    #print(document[0].metadata)
    save_to_chroma(chunks)    

if __name__ == "__main__":
    main()