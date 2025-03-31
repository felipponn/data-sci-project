import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def save_embeddings(docs_dict, queries_dict):
    """
    Sal

    Args:
        docs_dict (dict): {doc_id: texto documento}
        queries_dict (dict): {query_id: texto query}
    """
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    # Carregar modelo pré-treinado SBERT (rápido e eficiente)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Computar embeddings em lote (mais eficiente)
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

    # Create a dictionary with embeddings and IDs
    embedding_doc_data = {
        "doc_ids": doc_ids,  # List of document IDs
        "doc_embeddings": doc_embeddings.cpu(),  # Move to CPU before saving
    }
    embedding_query_data = {
        "query_ids": query_ids,  # List of query IDs
        "query_embeddings": query_embeddings.cpu(),  # Move to CPU before saving
    }

    torch.save(embedding_doc_data, "../data/embeddings/doc_embeddings.pt")
    torch.save(embedding_query_data, "../data/embeddings/query_embeddings.pt")

