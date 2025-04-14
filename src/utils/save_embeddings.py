import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

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

def save_embeddings_choose(docs_dict, queries_dict, model='bert'):
    """
    Salva os pesos dos embeddings de documentos e queries em arquivos .pt.

    Args:
        docs_dict (dict): {doc_id: texto documento}
        queries_dict (dict): {query_id: texto query}
        model (str): Nome do modelo a ser usado para embeddings. Padrão é 'bert'.
    """
    doc_ids, doc_texts = dict_to_list(docs_dict)
    query_ids, query_texts = dict_to_list(queries_dict)

    if model == 'bert':

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

        torch.save(embedding_doc_data, "../data/embeddings/bert_doc_embeddings.pt")
        torch.save(embedding_query_data, "../data/embeddings/bert_query_embeddings.pt")
    
    elif model == 'tevatron':
        tokenizer = AutoTokenizer.from_pretrained("castorini/mdpr-question_encoder")
        query_encoder = AutoModel.from_pretrained("castorini/mdpr-question_encoder")
        doc_encoder = AutoModel.from_pretrained("castorini/mdpr-passage_encoder")

        with torch.no_grad():
            doc_inputs = tokenizer(doc_texts,padding=True, truncation=True, return_tensors='pt')
            doc_outputs = doc_encoder(**doc_inputs)
            doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]  # CLS token

            query_inputs = tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt')
            query_outputs = query_encoder(**query_inputs)
            query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Create a dictionary with embeddings and IDs
        embedding_doc_data = {
            "doc_ids": doc_ids,  # List of document IDs
            "doc_embeddings": doc_embeddings.cpu(),  # Move to CPU before saving
        }
        embedding_query_data = {
            "query_ids": query_ids,  # List of query IDs
            "query_embeddings": query_embeddings.cpu(),  # Move to CPU before saving
        }

        torch.save(embedding_doc_data, "../data/embeddings/tevatron_doc_embeddings.pt")
        torch.save(embedding_query_data, "../data/embeddings/tevatron_query_embeddings.pt")

    else:
        raise ValueError("Modelo desconhecido. Use 'bert' ou 'tevatron'.")
