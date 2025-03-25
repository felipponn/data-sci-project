from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from tqdm import tqdm
import time

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def faiss_tevatron_search(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando Tevatron + FAISS para recuperação densa eficiente.
    
    Args:
        docs_dict: {doc_id: texto do documento}
        queries_dict: {query_id: texto da query}
        K: número de documentos a retornar por query

    Returns:
        dict: {query_id: [(doc_id, score), ...]}
        float: tempo de execução em segundos
    """
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    tokenizer_q = AutoTokenizer.from_pretrained("castorini/mdpr-question_encoder")
    tokenizer_d = AutoTokenizer.from_pretrained("castorini/mdpr-passage_encoder")
    
    query_encoder = AutoModel.from_pretrained("castorini/mdpr-question_encoder")
    doc_encoder = AutoModel.from_pretrained("castorini/mdpr-passage_encoder")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_encoder.to(device).eval()
    doc_encoder.to(device).eval()

    torch.set_grad_enabled(False)

    start_time = time.time()

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

    execution_time = time.time() - start_time
    return results, execution_time
