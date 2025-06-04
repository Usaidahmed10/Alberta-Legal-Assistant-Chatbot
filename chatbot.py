from config.settings import config
from preprocessing_pdfs.extract_text import extract_all_pdfs
from preprocessing_pdfs.clean_text import clean_all_texts
from preprocessing_pdfs.chunk_text import chunk_pdf_text
from metadata_embedding.format_chunks import format_metadata_chunks
from metadata_embedding.embed_chunks import embed_and_store_chunks
from retrieval.retrieve_documents import get_top_documents
from retrieval.retrieve_chunks import get_relevant_chunks
from llm_response.prompt_template import build_prompt
from llm_response.generate_response import get_final_answer

def run_pipeline(user_query):
    # 1. First-stage metadata RAG
    top_docs = get_top_documents(user_query)

    # 2. Load + clean + chunk full PDFs
    all_chunks = []
    for doc in top_docs:
        text = clean_all_texts(doc)
        chunks = chunk_pdf_text(text)
        all_chunks.extend(chunks)

    # 3. Second-stage fine retrieval
    relevant_chunks = get_relevant_chunks(user_query, all_chunks)

    # 4. Final LLM response
    prompt = build_prompt(user_query, relevant_chunks)
    response = get_final_answer(prompt)
    return response

if __name__ == "__main__":
    print(run_pipeline("What are the rights of tenants under the Residential Tenancies Act?"))
