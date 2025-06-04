import pandas as pd

def format_metadata_chunks(excel_path: str, max_rows: int = None):
    df = pd.read_excel(excel_path)
    if max_rows:
        df = df.head(max_rows)

    chunks = []

    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        keywords = str(row["keywords"]).strip()
        summary = str(row["summary"]).strip()

        # Optional: Reformat keywords into a sentence
        keyword_sentence = ", ".join([kw.strip() for kw in keywords.split(',')])
        meta_paragraph = (
            f"{title} is a legal document concerning {keyword_sentence}. "
            f"It primarily addresses the following: {summary}"
        )

        chunks.append({
            "doc_id": title,
            "text": meta_paragraph
        })
    return chunks

