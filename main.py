import os
import pandas as pd
import fitz  # For reading PDF
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load PDF or CSV
def load_document(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        text = df.to_string()
    else:
        raise ValueError("Unsupported file type.")
    return [Document(page_content=text)]

# Load fine-tuned model pipeline
def load_finetuned_llm():
    model_path = "./phi2-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, return_full_text=False)
    return pipe

# Build retriever
def build_retriever(docs):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    return retriever

# Main app loop
def main():
    file_path = input("Enter the path to your PDF or CSV file: ")
    print("Loading document...")
    docs = load_document(file_path)

    print("Setting up retrieval system...")
    retriever = build_retriever(docs)
    pipe = load_finetuned_llm()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            print("⚠️ No relevant documents found.")
            continue

        context = retrieved_docs[0].page_content.strip()
        print("\n🔎 Context Preview:\n", context[:500])

        prompt = f"<s>{context}\n{query}</s>"

        response = pipe(prompt)[0]["generated_text"]

        print("\n🧠 Answer:\n", response.strip().replace("\n", " "))


if __name__ == "__main__":
    main()
