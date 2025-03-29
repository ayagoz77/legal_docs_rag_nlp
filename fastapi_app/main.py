import os
import yaml
from fastapi import FastAPI, HTTPException
from groq import Groq
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from deep_translator import GoogleTranslator
import spacy
from dotenv import load_dotenv

load_dotenv()
# Load configuration from YAML file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Load FAISS index and sentence mapping
def load_faiss_index(index_path: str, mapping_path: str, all_laws_path: str):
    index = faiss.read_index(index_path)
    df = pd.read_csv(mapping_path)
    df_all = pd.read_csv(all_laws_path)
    return index, df, df_all

# Load embedding model
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

# Load reranker model
def load_reranker_model(model_name: str):
    return CrossEncoder(model_name)

# Load spaCy model
def load_ner_model(model_name: str):
    return spacy.load(model_name)

# Initialize components
index, sentence_mapping, all_laws = load_faiss_index(
    config["faiss_index_path"], config["sentence_mapping_path"], config["all_laws_path"]
)
reranker = load_reranker_model(config["reranker_model"])
embedding_model = load_embedding_model(config["embedding_model"])
nlp = load_ner_model(config["ner_model"])
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to filter relevant words based on dependency parsing
def extract_relevant_info(query: str) -> str:
    doc = nlp(query)
    ordered_tokens = []
    dep_order = ["det", "obl", "nsubj", "ROOT", "csubj", "amod"]
    
    for token in doc:
        if token.dep_ in dep_order:
            ordered_tokens.append(token.text)
    
    return " ".join(ordered_tokens)

def search_faiss(query: str, k: int = None):
    """Retrieve top-k candidates using FAISS and rerank them."""
    k = k or config["default_top_k"]
    query_vector = embedding_model.encode(["query: " + query])
    query_vector = np.array(query_vector, dtype=np.float32)

    # FAISS search
    D, I = index.search(query_vector, k)
    
    # Retrieve sentences and their indices from sentence_mapping
    candidates = [
        (
            sentence_mapping.loc[i, "val"],  # Get mapped sentence
            sentence_mapping.loc[i, "law_no"],
            sentence_mapping.loc[i, "article_no"],
            i  # Index in sentence_mapping
        )
        for i in I[0] if i in sentence_mapping.index
    ]

    # Prepare query-document pairs for reranking
    pairs = [(query, text) for text, _, _, _ in candidates]

    # Rerank results
    scores = reranker.predict(pairs)  # Get relevance scores
    ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Extract the best-matched law_no and article_no
    best_match = ranked_results[0][0] if ranked_results else None

    if best_match:
        best_law_no, best_article_no = best_match[1], best_match[2]

        # Retrieve the final law text from all_laws
        final_context = all_laws[
            (all_laws.law_no == best_law_no) &
            (all_laws.article_no == best_article_no)
        ].sentence.values

        return final_context[0] if len(final_context) > 0 else ""
    
    return ""

def formalize_query(query: str):
    prompt = f"""
    Переформулируй следующий вопрос в строго в формате "Кто что делать кому/чему?":
    {query}  

    Не включай другие вопросы кроме указанных в формате, отправь только измененное предложение.
    Не давай ответ на вопрос.  

    Результат:
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config["llm_model_formalizing"],
    )
    return chat_completion.choices[0].message.content

def generate_llm_response(context: str, query: str):
    prompt = f"""
    Контекст:
    {context}
    
    Вопрос:
    {query}
    
    Ответь на вопрос, основываясь на приведённом контексте. Используй только русские слова, убери все странные символы.  
    Если контекста недостаточно для ответа, дай разумный ответ, не упоминая, что контекста не хватает.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config["llm_model"],
    )
    return chat_completion.choices[0].message.content

def translate_to_english(text: str) -> str:
    return GoogleTranslator(source="auto", target="en").translate(text)

def translate_to_russian(text: str) -> str:
    return GoogleTranslator(source="auto", target="ru").translate(text)

app = FastAPI()

@app.get("/query/")
def query_law(query: str):
    try:
        formalized_query = query
        retrieved_sentences = search_faiss(formalized_query)
        context = retrieved_sentences
        # translated_context = translate_to_english(context)
        response = generate_llm_response(context, query)
        # translated_response = translate_to_russian(response)
        return {"query": query, "translated_query": context, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
