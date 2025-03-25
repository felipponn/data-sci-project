from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def tevatron_search(docs_dict, queries_dict, K=10):
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    tokenizer = AutoTokenizer.from_pretrained("castorini/mdpr-question_encoder")
    query_encoder = AutoModel.from_pretrained("castorini/mdpr-question_encoder")
    doc_encoder = AutoModel.from_pretrained("castorini/mdpr-passage_encoder")

    # Encode documentos
    with torch.no_grad():
        doc_inputs = tokenizer(doc_texts, padding=True, truncation=True, return_tensors='pt')
        doc_outputs = doc_encoder(**doc_inputs)
        doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]  # CLS token

    results = {}
    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        with torch.no_grad():
            q_inputs = tokenizer(qtext, return_tensors='pt', truncation=True, padding=True)
            q_outputs = query_encoder(**q_inputs)
            q_embedding = q_outputs.last_hidden_state[:, 0, :]  # CLS token

            # Similaridade por produto escalar
            scores = torch.matmul(doc_embeddings, q_embedding.squeeze().T).squeeze().numpy()

            top_k_idx = np.argsort(scores)[::-1][:K]
            results[qid] = [(doc_ids[idx], scores[idx]) for idx in top_k_idx]

    return results, None  # ou adicione time se quiser medir
