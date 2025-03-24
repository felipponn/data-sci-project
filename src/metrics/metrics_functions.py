import math

def precision_at_k(predicted_docs, relevant_docs, k=10):
    """
    Calcula Precision@K para uma query individual.
      - predicted_docs: lista (ou tupla) dos doc_ids retornados (em ordem do ranking).
      - relevant_docs: conjunto ou lista dos doc_ids relevantes.
    """
    # Se quiser garantir que a lista de predicted_docs tem pelo menos k elementos:
    top_k_docs = predicted_docs[:k]
    # Conta quantos desses top_k estão em relevant_docs
    hits = len(set(top_k_docs).intersection(set(relevant_docs)))
    # Precision@K = hits / K
    return hits / k

def mean_precision_at_k(results_dict, qrels_dict, k=10):
    """
    Calcula a média de Precision@K para um conjunto de queries.
    - results_dict: ex.: { qid: [(doc_id, score), ...], ...}
    - qrels_dict:   ex.: { qid: [doc_1, doc_2, ...], ...}
    """
    precisions = []
    for qid, ranked_docs_scores in results_dict.items():
        ranked_docs = [doc_id for doc_id, _ in ranked_docs_scores]
        # Caso não haja docs relevantes, define a precisão como 0 (ou ignore)
        if qid not in qrels_dict or len(qrels_dict[qid]) == 0:
            continue
        p_at_k = precision_at_k(ranked_docs, qrels_dict[qid], k)
        precisions.append(p_at_k)
    return sum(precisions) / len(precisions) if precisions else 0.0

def recall_at_k(predicted_docs, relevant_docs, k=10):
    """
    Calcula Recall@K para uma query individual.
      - predicted_docs: lista (ou tupla) dos doc_ids retornados (em ordem do ranking).
      - relevant_docs: conjunto ou lista dos doc_ids relevantes.
    """
    top_k_docs = predicted_docs[:k]
    hits = len(set(top_k_docs).intersection(set(relevant_docs)))
    total_relevantes = len(relevant_docs)
    # Se não houver documentos relevantes, retorna 0
    return hits / total_relevantes if total_relevantes > 0 else 0.0

def mean_recall_at_k(results_dict, qrels_dict, k=10):
    """
    Calcula a média de Recall@K para um conjunto de queries.
    - results_dict: ex.: { qid: [(doc_id, score), ...], ...}
    - qrels_dict:   ex.: { qid: [doc_1, doc_2, ...], ...}
    """
    recalls = []
    for qid, ranked_docs_scores in results_dict.items():
        ranked_docs = [doc_id for doc_id, _ in ranked_docs_scores]
        # Caso não haja docs relevantes para a query, ignora
        if qid not in qrels_dict or len(qrels_dict[qid]) == 0:
            continue
        recalls.append(recall_at_k(ranked_docs, qrels_dict[qid], k))
    return sum(recalls) / len(recalls) if recalls else 0.0


def average_precision(predicted_docs, relevant_docs, k=10):
    """
    Calcula Average Precision (AP) para UMA query, considerando até K resultados.
    A AP leva em conta a posição em que cada doc relevante aparece.
    """
    relevant_docs = set(relevant_docs)
    # Se não há docs relevantes, retorna 0.
    if not relevant_docs:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(predicted_docs[:k], start=1):
        if doc_id in relevant_docs:
            hits += 1
            # Precisão no rank i
            precision_i = hits / i
            sum_precisions += precision_i
    
    return sum_precisions / len(relevant_docs)

def mean_average_precision_at_k(results_dict, qrels_dict, k=10):
    """
    Calcula Mean Average Precision (MAP@K) para o conjunto de queries.
    """
    ap_scores = []
    for qid, ranked_docs_scores in results_dict.items():
        ranked_docs = [doc_id for doc_id, _ in ranked_docs_scores]
        # Se a query não tiver docs relevantes, ignora ou trata como AP=0
        if qid not in qrels_dict or len(qrels_dict[qid]) == 0:
            continue
        ap = average_precision(ranked_docs, qrels_dict[qid], k)
        ap_scores.append(ap)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def reciprocal_rank(predicted_docs, relevant_docs):
    """
    Calcula o Reciprocal Rank (RR) para uma única query.
    O RR é definido como 1 / rank_do_primeiro_documento_relevante.
    Se não houver documento relevante na lista, o RR será 0.
    
    :param predicted_docs: lista (ou tupla) dos doc_ids retornados (em ordem do ranking).
    :param relevant_docs: conjunto ou lista dos doc_ids relevantes.
    :return: valor do Reciprocal Rank (float).
    """
    relevant_set = set(relevant_docs)
    for i, doc_id in enumerate(predicted_docs, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(results_dict, qrels_dict):
    """
    Calcula o MRR (Mean Reciprocal Rank) para um conjunto de queries.
    
    :param results_dict: dicionário no formato { qid: [(doc_id, score), ...], ... }
    :param qrels_dict:   dicionário no formato { qid: [doc_1, doc_2, ...], ... }
    :return: valor do MRR (float).
    """
    rr_scores = []
    for qid, ranked_docs_scores in results_dict.items():
        ranked_docs = [doc_id for doc_id, _ in ranked_docs_scores]
        
        # Verifica se há documentos relevantes para a query
        if qid in qrels_dict and len(qrels_dict[qid]) > 0:
            rr = reciprocal_rank(ranked_docs, qrels_dict[qid])
            rr_scores.append(rr)
    
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

def dcg_at_k(predicted_docs, relevant_docs, k=10):
    """
    Calcula o DCG (Discounted Cumulative Gain) até a posição K para uma única query.
    Para cada documento relevante, a contribuição é 1 / log2(posição+1).
    
    :param predicted_docs: lista de doc_ids retornados (ordenados).
    :param relevant_docs: conjunto ou lista de doc_ids relevantes.
    :param k: profundidade de corte (top-k).
    :return: valor de DCG (float).
    """
    relevant_set = set(relevant_docs)
    dcg = 0.0
    for i, doc_id in enumerate(predicted_docs[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def ndcg_at_k(predicted_docs, relevant_docs, k=10):
    """
    Calcula o NDCG (Normalized Discounted Cumulative Gain) até a posição K
    para uma única query.
    
    :param predicted_docs: lista de doc_ids retornados (ordenados).
    :param relevant_docs: conjunto ou lista de doc_ids relevantes.
    :param k: profundidade de corte (top-k).
    :return: valor de NDCG (float).
    """
    # DCG calculado com a ordem predita
    dcg_value = dcg_at_k(predicted_docs, relevant_docs, k)
    
    # Cálculo do IDCG (melhor DCG possível ordenando todos os relevantes no topo)
    ideal_ranks = min(len(relevant_docs), k)
    idcg = 0.0
    for i in range(1, ideal_ranks + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    return dcg_value / idcg if idcg > 0 else 0.0

def mean_ndcg_at_k(results_dict, qrels_dict, k=10):
    """
    Calcula a média de NDCG@K para um conjunto de queries.
    
    :param results_dict: dicionário no formato { qid: [(doc_id, score), ...], ... }
    :param qrels_dict:   dicionário no formato { qid: [doc_1, doc_2, ...], ... }
    :param k: profundidade de corte (top-k).
    :return: valor médio de NDCG@K (float).
    """
    ndcg_scores = []
    for qid, ranked_docs_scores in results_dict.items():
        ranked_docs = [doc_id for doc_id, _ in ranked_docs_scores]
        
        # Verifica se há documentos relevantes para a query
        if qid in qrels_dict and len(qrels_dict[qid]) > 0:
            ndcg_value = ndcg_at_k(ranked_docs, qrels_dict[qid], k)
            ndcg_scores.append(ndcg_value)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
