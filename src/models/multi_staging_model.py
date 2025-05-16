from tqdm import tqdm
import time
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import pickle

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
def secondary_stage(docs_dict, queries_dict, top_k_results, set_docs, K=10, load_encodings=True):
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
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
    if load_encodings == False:
        doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    else:
        # Carregar os embeddings pré-computados
        print("Carregando embeddings pré-computados.")
        # loaded_data_doc = "../data/embeddings/doc_embeddings.pt" with pickle
        loaded_data_doc = pickle.load(open("../data/embeddings/doc_embeddings.pt", "rb"))
        

        doc_embeddings = loaded_data_doc["embeddings"]
        doc_ids_loaded = loaded_data_doc["doc_ids"]
        
        #filtrar os indices loadded_data_doc["doc_ids"] para os ids que foram recuperados na primeira fase
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids_loaded)}

    results = {}
    for qid, query_emb in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
        # Get the doc_ids for the current query
        filtered_doc_ids = top_k_results[qid]
        
        # Convert these doc_ids to indices using the mapping:
        if load_encodings:
            filtered_doc_indices = [doc_id_to_index[doc_id] for doc_id in filtered_doc_ids]
        else:
            # If not loading pre-computed encodings, build a mapping for the current ordering
            local_doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            filtered_doc_indices = [local_doc_id_to_index[doc_id] for doc_id in filtered_doc_ids]
        
        # Now, use the integer indices to get the embeddings:
        filtered_doc_embeddings = doc_embeddings[filtered_doc_indices]
        filtered_doc_embeddings = filtered_doc_embeddings.cpu().to(query_emb.device)

        # Compute cosine similarity:
        cos_scores = util.cos_sim(query_emb, filtered_doc_embeddings)[0]

        # Get top K results:
        top_results = torch.topk(cos_scores, k=min(K, len(filtered_doc_indices)))

        results[qid] = [
            (filtered_doc_ids[idx], float(score))
            for score, idx in zip(top_results.values, top_results.indices)
        ]
        execution_time = time.time() - start_time

    return results, execution_time