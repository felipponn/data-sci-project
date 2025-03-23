import random
from tqdm import tqdm
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
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

import time
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def dict_to_list(d):
    """Converte dicionário {id: texto} em duas listas paralelas [ids], [textos]."""
    ids = list(d.keys())
    textos = list(d.values())
    return ids, textos

def tfidf_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando TF-IDF e similaridade de cosseno em lote, retornando os Top K documentos.
    
    Args:
        docs_dict (dict): {doc_id: texto_do_documento}
        queries_dict (dict): {query_id: texto_da_query}
        K (int): número de documentos a retornar por query (default=10)
    
    Returns:
        dict: {query_id: [(doc_id, score), ...]}, para cada query.
        float: tempo de execução em segundos
    """
    # Passo 1: Obter listas de ids e textos
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    start_time = time.time()

    # Passo 2: Vetorizar documentos (ajuste do vocabulário)
    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(doc_texts)  # shape: [num_docs x num_features]

    # Passo 3: Vetorizar todas as queries de uma só vez (mesmo vocabulário)
    tfidf_queries = vectorizer.transform(query_texts) # shape: [num_queries x num_features]

    # Passo 4: Calcular similaridade em lote usando produto de matrizes
    # Como o TF-IDF por padrão já é normalizado em L2,
    # o produto escalar tfidf_queries * tfidf_docs.T corresponde à cosseno-similaridade.
    similarity_matrix = tfidf_queries.dot(tfidf_docs.T)  # shape: [num_queries x num_docs]

    top_k_results = {}
    for i in tqdm(range(similarity_matrix.shape[0]), total=similarity_matrix.shape[0]):
        # Extraindo a i-ésima linha
        row_scores = similarity_matrix[i].toarray().ravel()

        top_k_idx = np.argpartition(row_scores, -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(-row_scores[top_k_idx])]

        results = [(doc_ids[idx], row_scores[idx]) for idx in top_k_idx]
        top_k_results[query_ids[i]] = results

    execution_time = time.time() - start_time
    
    return top_k_results, execution_time



def bm25_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando BM25 com seleção parcial para o Top K.
    
    Args:
        docs_dict: dicionário com {doc_id: texto do documento}
        queries_dict: dicionário com {query_id: texto da query}
        K: número de documentos a retornar por query (default=10)
    
    Returns:
        dict: dicionário com resultados no formato {qid: [(doc_id, score), ...]}
        float: tempo de execução em segundos
    """
    # Passo 1: Obter listas de IDs e textos
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    start_time = time.time()

    # Passo 2: Tokenizar cada documento
    tokenized_docs = [doc.split() for doc in doc_texts]

    # Passo 3: Criar o objeto BM25Okapi
    bm25 = BM25Okapi(tokenized_docs)

    # Dicionário de resultados
    top_k_results = {}

    # Passo 4: Para cada query, calcular scores e pegar top K docs (usando seleção parcial)
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        query_tokens = qtext.split()
        
        # Obtém scores de BM25 para cada documento
        scores = bm25.get_scores(query_tokens)
        
        # Seleciona índices do Top K (sem ordenar toda a lista)
        idxs_top_k = np.argpartition(scores, -K)[-K:]
        
        # Ordena apenas os K índices selecionados
        idxs_top_k = idxs_top_k[np.argsort(scores[idxs_top_k])[::-1]]
        
        # Mapeia índices de volta aos doc_ids
        top_docs = [(doc_ids[idx], scores[idx]) for idx in idxs_top_k]
        top_k_results[qid] = top_docs

    execution_time = time.time() - start_time
    
    return top_k_results, execution_time
