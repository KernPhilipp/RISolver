import sys
import os
import math
import requests
import faiss
import concurrent.futures
import re
import time

from datetime import date
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

print(">>> Alle Bibliotheken importiert und bereit.")

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:1b"
INDEX_PATH = "faiss_index.faiss"
API_URL = "https://data.bka.gv.at/ris/api/v2.6/Landesrecht"


def build_or_load_index():
    print(">>> Building FAISS index, please wait…")
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, OllamaEmbeddings(model=EMBEDDING_MODEL),
                                allow_dangerous_deserialization=True)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    session = requests.Session()

    def extract_pdf_url(ref) -> str | None:
        lrKons = (
            ref.get("Data", {})
            .get("Metadaten", {})
            .get("Landesrecht", {})
            .get("LrKons", {})
        )
        html_url = lrKons.get("GesamteRechtsvorschriftUrl")
        if not html_url:
            return None
        try:
            resp = session.get(html_url, timeout=60)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            a_pdf = soup.find(
                "a",
                href=re.compile(r"\.pdf$"),
                title=lambda t: t and "PDF-Dokument" in t
            )
            if not a_pdf or not a_pdf.has_attr("href"):
                return None

            pdf_path = a_pdf["href"]
            return urljoin("https://www.ris.bka.gv.at", pdf_path)
        except Exception:
            return None

    def load_pdf_pages(pdf_url: str) -> list[tuple[str, str]]:
        try:
            loader = PyPDFLoader(pdf_url)
            docs = loader.load()
            return [(pdf_url, doc.page_content) for doc in docs]
        except Exception:
            return []

    heute = date.today().isoformat()
    params = {
        "Applikation": "LrKons",
        "Bundesland_SucheInSalzburg": "true",
        "Sortierung_SortDirection": "Descending",
        "DokumenteProSeite": "Ten",
        "Fassung_FassungVom": heute
    }
    resp = session.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    hits = int(data["OgdSearchResult"]["OgdDocumentResults"]["Hits"]["#text"])
    page_size = int(data["OgdSearchResult"]["OgdDocumentResults"]["Hits"]["@pageSize"])
    total = math.ceil(hits / page_size)

    all_refs = []
    # for page in range(1, total+1):
    for page in range(1, 2):
        params["Seitennummer"] = page
        resp = session.get(API_URL, params=params)
        resp.raise_for_status()
        docs = resp.json()["OgdSearchResult"]["OgdDocumentResults"]["OgdDocumentReference"]
        all_refs.extend(docs if isinstance(docs, list) else [docs])

    pdf_urls = {u for ref in all_refs if (u := extract_pdf_url(ref))}

    all_sections = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load_pdf_pages, url) for url in pdf_urls]
        for fut in concurrent.futures.as_completed(futures):
            all_sections.extend(fut.result() or [])

    para_docs = []
    for url, text in all_sections:
        cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
        for para in cleaned.split("\n\n"):
            if para.strip():
                para_docs.append(Document(
                    page_content=f"Quelle: {url}\n{para.strip()}",
                    metadata={"source_url": url}
                ))

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1200,
        chunk_overlap=0
    )
    chunked_docs = splitter.split_documents(para_docs)

    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    vector_store.save_local(INDEX_PATH)
    return vector_store


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Du bist ein hilfreicher Assistent, der Informationen aus dem konsolidierten Landesrecht Salzburgs bereitstellt.
Jeder Auszug im Kontext beginnt mit einer Zeile `Quelle: URL`.

=== Auszüge ===
{context}

=== Frage ===
{question}

=== Antwort ===
Bitte beantworte die Frage klar und präzise, erkläre schwierige Begriffe einfach, und gib am Ende die Quelle an.
"""
)


def ask_rag(question: str) -> str:
    idx = build_or_load_index()
    retriever = idx.as_retriever(search_kwargs={"k": 10})
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0, max_tokens=1024)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain.run(question)


def main():
    if len(sys.argv) != 2:
        print("Usage: rag_cli.py \"Deine Frage hier\"", file=sys.stderr)
        sys.exit(1)
    question = sys.argv[1]
    answer = ask_rag(question)
    print(answer)


if __name__ == "__main__":
    main()
