from tqdm import tqdm
import time
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

# Primeiro estágio, modelo simples focado em recall (bm25, tfidf)
# retorna os docs mais relevantes unicamente
# reorna uma estrutura pareando query_id com os ids dos documentos mais relevantes

def primary_stage(docs_dict, queries_dict, K=1000):
    """
    Realiza a busca de documentos usando um modelo simples (BM25 ou TF-IDF).

    Args:
        docs_dict (dict): Dicionário de documentos com IDs e textos.
        queries_dict (dict): Dicionário de consultas com IDs e textos.
        K (int): Número de documentos a serem recuperados para cada consulta.

    Returns:
        dict: Dicionário com os IDs dos documentos recuperados com os ids das consultas.
        set: conjunto de ids de documentos recuperados.
        float: Tempo total de execução.
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

    set_docs = set()

    top_k_results = {}
    for i in tqdm(range(similarity_matrix.shape[0]), total=similarity_matrix.shape[0]):
        # Extraindo a i-ésima linha
        row_scores = similarity_matrix[i].toarray().ravel()

        top_k_idx = np.argpartition(row_scores, -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(-row_scores[top_k_idx])]

        results = [(doc_ids[idx]) for idx in top_k_idx]
        top_k_results[query_ids[i]] = results
        set_docs.update(results)

    execution_time = time.time() - start_time

    return top_k_results, set_docs, execution_time


# Usando modelo de precisão consideramos apenas os documentos que estão entre os K mais relevantes para cada consulta
# e retornamos os ids dos documentos mais relevantes para cada consulta
def secondary_stage(docs_dict, queries_dict, top_k_results, set_docs, K=10):
    """
    Realiza a busca de documentos usando um modelo mais complexo (BERT ou similar).

    Args:
        docs_dict (dict): Dicionário de documentos com IDs e textos.
        queries_dict (dict): Dicionário de consultas com IDs e textos.
        K (int): Número de documentos a serem recuperados para cada consulta.

    Returns:
        dict: Dicionário com os IDs dos documentos recuperados e suas pontuações de similaridade.
        float: Tempo total de execução.
    """
    start_time = time.time()

    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    # Carregar modelo pré-treinado SBERT (rápido e eficiente)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Computar embeddings em lote (mais eficiente)
    # Encoda apenas os documentos que foram recuperados na primeira fase

    doc_texts = [docs_dict[doc_id] for doc_id in set_docs]
    doc_ids = [doc_id for doc_id in set_docs]
    
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

    # Dicionário para armazenar resultados
    results = {}

    # Criar um dicionário de mapeamento de ID para índice na lista de embeddings
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    # Para cada query, calcular similaridade de cosseno
    for qid, query_emb in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
        # Obter os índices correspondentes dos documentos relevantes
        filtered_indices = [doc_id_to_index[doc_id] for doc_id in top_k_results[qid]]

        # Filtrar os embeddings dos documentos mais relevantes
        filtered_doc_embeddings = doc_embeddings[filtered_indices]  # Agora indexamos corretamente
        
        # Calcular similaridade entre query e todos documentos
        cos_scores = util.cos_sim(query_emb, filtered_doc_embeddings)[0]

        # Obter top K documentos mais similares
        top_results = torch.topk(cos_scores, k=K)

        results[qid] = [
            (top_k_results[qid][idx], float(score))
            for score, idx in zip(top_results.values, top_results.indices)
        ]
    execution_time = time.time() - start_time

    return results, execution_time