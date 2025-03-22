import random
from tqdm import tqdm
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def random_search(docs_dict, queries_dict, K=10):
    """
    Realiza uma busca aleatória, retornando K documentos aleatórios para cada query.
    
    Args:
        docs_dict: dicionário com {doc_id: texto do documento}
        queries_dict: dicionário com {query_id: texto da query}
        K: número de documentos a retornar por query (default=10)
    
    Returns:
        dict: dicionário com resultados no formato {qid: [(doc_id, score), ...]}
        float: tempo de execução em segundos
    """
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    # Para guardar resultados
    top_k_results = {}

    # Medir tempo de execução
    start_time = time.time()

    # Para cada query
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        random_doc_indices = random.sample(range(len(doc_ids)), K)
        
        # Retorna doc_id e score simbólico (0.0) para cada documento
        top_docs = [(doc_ids[idx], 0.0) for idx in random_doc_indices]
        top_k_results[qid] = top_docs

    execution_time = time.time() - start_time
    
    return top_k_results, execution_time

def tfidf_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando TF-IDF e similaridade de cosseno.
    
    Args:
        docs_dict: dicionário com {doc_id: texto do documento}
        queries_dict: dicionário com {query_id: texto da query}
        K: número de documentos a retornar por query (default=10)
    
    Returns:
        dict: dicionário com resultados no formato {qid: [(doc_id, score), ...]}
        float: tempo de execução em segundos
    """
    # Passo 1: Obter listas dos textos e ids
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    start_time = time.time()

    # Passo 2: Vetorizar documentos
    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(doc_texts)

    # Dicionário de resultados
    top_k_results = {}

    # Passo 3: Para cada query, calcular similaridade e pegar top N docs
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        tfidf_query = vectorizer.transform([qtext])  # transforma a query no mesmo espaço vetorial
        cosine_similarities = cosine_similarity(tfidf_query, tfidf_docs).flatten()
        top_doc_indices = np.argsort(cosine_similarities)[::-1][:K]  # top N índices

        # Mapear índices de volta aos doc_ids
        top_docs = [(doc_ids[idx], cosine_similarities[idx]) for idx in top_doc_indices]
        top_k_results[qid] = top_docs

    execution_time = time.time() - start_time
    
    return top_k_results, execution_time


def bm25_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando BM25.
    
    Args:
        docs_dict: dicionário com {doc_id: texto do documento}
        queries_dict: dicionário com {query_id: texto da query}
        K: número de documentos a retornar por query (default=10)
    
    Returns:
        dict: dicionário com resultados no formato {qid: [(doc_id, score), ...]}
        float: tempo de execução em segundos
    """
    # Passo 1: Obter listas dos textos e ids
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    start_time = time.time()

    # Passo 2: Tokenizar cada documento
    tokenized_docs = [doc.split() for doc in doc_texts]

    # Passo 3: Criar o objeto BM25Okapi
    bm25 = BM25Okapi(tokenized_docs)

    # Dicionário de resultados
    top_k_results = {}

    # Passo 4: Para cada query, calcular scores e pegar top K docs
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        query_tokens = qtext.split()
        
        # Obtém scores de BM25 para cada documento
        scores = bm25.get_scores(query_tokens)
        
        # Classifica os docs pelo score (maior -> menor)
        top_doc_indices = np.argsort(scores)[::-1][:K]
        
        # Mapeia índices de volta aos doc_ids
        top_docs = [(doc_ids[idx], scores[idx]) for idx in top_doc_indices]
        top_k_results[qid] = top_docs

    execution_time = time.time() - start_time
    
    return top_k_results, execution_time
