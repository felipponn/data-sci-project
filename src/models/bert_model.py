from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util

from tqdm import tqdm
import time
import torch
import numpy as np

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def encode_texts(texts, tokenizer, model, device, batch_size=32, desc="Encoding"):
    """Encoda textos em embeddings usando BERT.
    Args:
        texts (list): Lista de textos a serem codificados.
        tokenizer: Tokenizer do modelo BERT.
        model: Modelo BERT.
        device: Dispositivo (CPU ou GPU) para executar o modelo.
        batch_size (int): Tamanho do lote para processamento em lote.
        desc (str): Descrição para a barra de progresso.
        
    Returns:
        torch.Tensor: Tensor contendo os embeddings dos textos.
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[i : i + batch_size]
        tokenized = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            outputs = model(**tokenized)
        
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)

def bert(docs_dict, queries_dict, tokenizer, model, K=10, batch_size=32):
    """
    Realiza a busca de documentos usando o modelo BERT.

    Args:
        docs_dict (dict): Dicionário de documentos com IDs e textos.
        queries_dict (dict): Dicionário de consultas com IDs e textos.
        tokenizer: Tokenizer do modelo BERT.
        model: Modelo BERT.
        K (int): Número de documentos a serem recuperados para cada consulta.
        batch_size (int): Tamanho do lote para processamento em lote.

    Returns:
        dict: Dicionário com os IDs dos documentos recuperados e suas pontuações de similaridade.
        float: Tempo total de execução.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)
    
    start_time = time.time()

    # Encode all documents in batches
    doc_embeddings = encode_texts(doc_texts, tokenizer, model, device, batch_size, desc="Encoding documents")

    # Encode all queries in batches
    query_embeddings = encode_texts(query_texts, tokenizer, model, device, batch_size, desc="Encoding queries")

    # Compute cosine similarity in batches
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=1)  # Normalize for cosine similarity
    query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

    similarity_matrix = torch.mm(query_embeddings, doc_embeddings.T)  # Shape: (num_queries, num_docs)

    top_k_results = {}
    for i, qid in tqdm(enumerate(query_ids), total=len(query_ids), desc="Retrieving top-K docs", unit="query"):
        top_k_indices = torch.argsort(similarity_matrix[i], descending=True)[:K]
        top_k_results[qid] = [(doc_ids[idx], similarity_matrix[i, idx].item()) for idx in top_k_indices]

    execution_time = time.time() - start_time
    return top_k_results, execution_time

# Exemplo de uso:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# results, exec_time = bert(docs_dict, queries_dict, tokenizer, model, batch_size=32)




def bert_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando Sentence-BERT e similaridade de cosseno.

    Args:
        docs_dict (dict): {doc_id: texto documento}
        queries_dict (dict): {query_id: texto query}
        K (int): Número de documentos retornados por query

    Returns:
        dict: {query_id: [(doc_id, score), ...]}
        float: tempo total execução (segundos)
    """
    start_time = time.time()

    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    # Carregar modelo pré-treinado SBERT (rápido e eficiente)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Computar embeddings em lote (mais eficiente)
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)
    query_embeddings = model.encode(query_texts, convert_to_tensor=True)

    # Dicionário para armazenar resultados
    results = {}

    # Para cada query, calcular similaridade de cosseno
    for qid, query_emb in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
        # Calcular similaridade entre query e todos documentos
        cos_scores = util.cos_sim(query_emb, doc_embeddings)[0]

        # Obter top K documentos mais similares
        top_results = torch.topk(cos_scores, k=K)

        results[qid] = [
            (doc_ids[idx], float(score))
            for score, idx in zip(top_results.values, top_results.indices)
        ]

    execution_time = time.time() - start_time

    return results, execution_time