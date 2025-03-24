from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import time
import torch
import numpy as np

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def bert(docs_dict, queries_dict, K=10):
    """
    Realiza busca usando BERT e similaridade de cosseno em lote, retornando os Top K documentos.
    
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
    
    # Tokenizador
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Carregar modelo
    model = BertModel.from_pretrained('bert-base-uncased')

    # Medir tempo de execução
    start_time = time.time()

    top_k_results = {}

    # Para cada query
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):

        inputs = tokenizer(qtext, doc_texts, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)

        relevance_scores = torch.nn.functional.sigmoid(outputs.last_hidden_state[:, 0, :])
        relevance_scores = relevance_scores.detach().numpy().flatten()

        top_k_indices = np.argsort(relevance_scores)[::-1][:K]
        top_k_results[qid] = [(doc_ids[idx], relevance_scores[idx]) for idx in top_k_indices]
    
    execution_time = time.time() - start_time
    
    return top_k_results, execution_time
