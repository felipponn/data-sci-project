from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import time
import torch
import numpy as np

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def encode_texts(texts, tokenizer, model, device, batch_size=32, desc="Encoding"):
    """Encodes a list of texts into embeddings using BERT in batches with tqdm progress bar."""
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
    Performs document retrieval using BERT embeddings and cosine similarity.

    Args:
        docs_dict: {doc_id: doc_text}
        queries_dict: {query_id: query_text}
        tokenizer: Preloaded BERT tokenizer
        model: Preloaded BERT model
        K: Number of top documents to return per query (default=10)
        batch_size: Number of documents to encode per batch (default=32)

    Returns:
        dict: {query_id: [(doc_id, score), ...]}
        float: Execution time in seconds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)
    
    start_time = time.time()

    # Encode documents in batches
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

# Load model and tokenizer once
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# Example call
# results, exec_time = bert(docs_dict, queries_dict, tokenizer, model, batch_size=32)
