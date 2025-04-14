from tqdm import tqdm
import time
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import faiss

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
    
    if load_encodings == False:
        doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
        query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
    else:
        # Carregar os embeddings pré-computados
        print("Carregando embeddings pré-computados.")
        loaded_data_doc = torch.load("../data/embeddings/doc_embeddings.pt")
        loaded_data_query = torch.load("../data/embeddings/query_embeddings.pt")
        doc_embeddings = loaded_data_doc["doc_embeddings"]
        doc_ids = loaded_data_doc["doc_ids"]

        query_embeddings = loaded_data_query["query_embeddings"]
        query_ids = loaded_data_query["query_ids"]

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


def primary_stage_choose(docs_dict, queries_dict, K=1000,
                         model='tf-idf'):
    """
    Realiza a busca de documentos usando um modelo simples (BM25 ou TF-IDF).

    Args:
        docs_dict (dict): Dicionário de documentos com IDs e textos.
        queries_dict (dict): Dicionário de consultas com IDs e textos.
        K (int): Número de documentos a serem recuperados para cada consulta.
        model (str): modelo escolhido para fazer a busca.

    Returns:
        dict: Dicionário com os IDs dos documentos recuperados com os ids das consultas.
        set: conjunto de ids de documentos recuperados.
        float: Tempo total de execução.
    """
    # Passo 1: Obter listas de ids e textos
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    start_time = time.time()

    if model == 'bm25':
        # Passo 2: Tokenizar cada documento.
        tokenized_docs = [doc.split() for doc in doc_texts]

         # Passo 3: Criar o objeto BM25Okapi
        bm25 = BM25Okapi(tokenized_docs)

        # Dicionário de resultados
        top_k_results = {}
        set_docs = set()

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
            top_docs = [(doc_ids[idx]) for idx in idxs_top_k]
            top_k_results[qid] = top_docs
            set_docs.update(top_docs)

    else:
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


def secondary_stage_choose(docs_dict, queries_dict, top_k_results, set_docs, K=10, load_encodings=True,
                           model='bert'):
    """
    Realiza a busca de documentos usando um modelo mais complexo (BERT ou similar).

    Args:
        docs_dict (dict): Dicionário de documentos com IDs e textos.
        queries_dict (dict): Dicionário de consultas com IDs e textos.
        K (int): Número de documentos a serem recuperados para cada consulta.
        model (str): modelo de busca escolhido;

    Returns:
        dict: Dicionário com os IDs dos documentos recuperados e suas pontuações de similaridade.
        float: Tempo total de execução.
    """
    start_time = time.time()

    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    if model == 'tevatron':
        tokenizer = AutoTokenizer.from_pretrained("castorini/mdpr-question_encoder")
        query_encoder = AutoModel.from_pretrained("castorini/mdpr-question_encoder")
        doc_encoder = AutoModel.from_pretrained("castorini/mdpr-passage_encoder")

        doc_texts = [docs_dict[doc_id] for doc_id in set_docs]
        doc_ids = [doc_id for doc_id in set_docs]

        if load_encodings == False:
            # Encode docs
            with torch.no_grad():
                doc_inputs = tokenizer(doc_texts,padding=True, truncation=True, return_tensors='pt')
                doc_outputs = doc_encoder(**doc_inputs)
                doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]  # CLS token

                query_inputs = tokenizer(query_texts, return_tensors='pt', truncation=True, padding=True)
                query_outputs = query_encoder(**query_inputs)
                query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            # Carregar os embeddings pré-computados
            print("Carregando embeddings pré-computados.")
            loaded_data_doc = torch.load("../data/embeddings/tevatron_doc_embeddings.pt")
            loaded_data_query = torch.load("../data/embeddings/tevatron_query_embeddings.pt")
            doc_embeddings = loaded_data_doc["doc_embeddings"]
            doc_ids = loaded_data_doc["doc_ids"]

            query_embeddings = loaded_data_query["query_embeddings"]
            query_ids = loaded_data_query["query_ids"]

        results = {}

        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        for qid, qtext in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
            # Obter os índices correspondentes dos documentos relevantes
            filtered_indices = [doc_id_to_index[doc_id] for doc_id in top_k_results[qid]]

            # Filtrar os embeddings dos documentos mais relevantes
            filtered_doc_embeddings = doc_embeddings[filtered_indices]  # Agora indexamos corretamente

            # Calcular similaridade entre query e todos documentos
            cos_scores = util.cos_sim(qtext, filtered_doc_embeddings)[0]

            # Obter top K documentos mais similares
            top_results = torch.topk(cos_scores, k=K)

            results[qid] = [
                (top_k_results[qid][idx], float(score))
                for score, idx in zip(top_results.values, top_results.indices)
            ]
    
    elif model == 'faiss':
        tokenizer_q = AutoTokenizer.from_pretrained("castorini/mdpr-question_encoder")
        tokenizer_d = AutoTokenizer.from_pretrained("castorini/mdpr-passage_encoder")
        
        query_encoder = AutoModel.from_pretrained("castorini/mdpr-question_encoder")
        doc_encoder = AutoModel.from_pretrained("castorini/mdpr-passage_encoder")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_encoder.to(device).eval()
        doc_encoder.to(device).eval()

        torch.set_grad_enabled(False)

        # =======================
        # 1. Embedding dos documentos
        # =======================
        doc_inputs = tokenizer_d(doc_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
        doc_outputs = doc_encoder(**doc_inputs)
        doc_embeddings = doc_outputs.last_hidden_state[:, 0, :].cpu().numpy()

        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Produto escalar
        index.add(doc_embeddings)       # Adiciona os vetores

        # =======================
        # 2. Buscar queries no índice FAISS
        # =======================
        results = {}

        for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
            q_inputs = tokenizer_q(qtext, return_tensors='pt', truncation=True, padding=True, max_length=512)
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            q_output = query_encoder(**q_inputs)
            q_embedding = q_output.last_hidden_state[:, 0, :].cpu().numpy()

            scores, indices = index.search(q_embedding, K)  # (1 x K)
            top_docs = [(doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
            results[qid] = top_docs
    
    else:
        # Carregar modelo pré-treinado SBERT (rápido e eficiente)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Computar embeddings em lote (mais eficiente)
        # Encoda apenas os documentos que foram recuperados na primeira fase

        doc_texts = [docs_dict[doc_id] for doc_id in set_docs]
        doc_ids = [doc_id for doc_id in set_docs]
        
        if load_encodings == False:
            doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
            query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
        else:
            # Carregar os embeddings pré-computados
            print("Carregando embeddings pré-computados.")
            loaded_data_doc = torch.load("../data/embeddings/bert_doc_embeddings.pt")
            loaded_data_query = torch.load("../data/embeddings/bert_query_embeddings.pt")
            doc_embeddings = loaded_data_doc["doc_embeddings"]
            doc_ids = loaded_data_doc["doc_ids"]

            query_embeddings = loaded_data_query["query_embeddings"]
            query_ids = loaded_data_query["query_ids"]

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