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
