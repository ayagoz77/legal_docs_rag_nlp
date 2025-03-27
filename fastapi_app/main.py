import os
import yaml
from fastapi import FastAPI, HTTPException
from groq import Groq
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import spacy

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

# Load spaCy model
def load_ner_model(model_name: str):
    return spacy.load(model_name)

# Initialize components
index, sentence_mapping, all_laws = load_faiss_index(
    config["faiss_index_path"], config["sentence_mapping_path"], config["all_laws_path"]
)
model = load_embedding_model(config["embedding_model"])
nlp = load_ner_model(config["ner_model"])
client = Groq(api_key=os.environ.get("GROQ_API_KEY", config["groq_api_key"]))

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
    k = config["default_top_k"]
    query_vector = model.encode(["query: " + query])
    query_vector = np.array(query_vector, dtype=np.float32)
    D, I = index.search(query_vector, k)
    print(sentence_mapping.loc[I[0], 'law_no'].tolist()[0], sentence_mapping.loc[I[0], 'article_no'].tolist()[0])
    print(type(all_laws[
        (all_laws.law_no == sentence_mapping.loc[I[0], 'law_no'].tolist()[0]) & 
        (all_laws.article_no == sentence_mapping.loc[I[0], 'article_no'].tolist()[0])
    ].sentence.tolist()[0]))
    return all_laws[
        (all_laws.law_no == sentence_mapping.loc[I[0], 'law_no'].tolist()[0]) & 
        (all_laws.article_no == sentence_mapping.loc[I[0], 'article_no'].tolist()[0])
    ].sentence.tolist()[0]

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
    Context:
    {context}
    
    Question:
    {query}
    
    Answer the question depending on the context above, use only english words, remove all strange symbols, 
    if context is not enough to answer the question, please provide a reasonable answer 
    (not mention that context is not enough).:
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
        formalized_query = extract_relevant_info(query)
        print(formalized_query)
        retrieved_sentences = search_faiss(formalized_query)
        context = retrieved_sentences
        translated_context = translate_to_english(context)
        response = generate_llm_response(translated_context, query)
        translated_response = translate_to_russian(response)
        return {"query": query, "translated_query": context, "response": translated_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
