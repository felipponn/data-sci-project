import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import pickle
import os
import sys
# run this script from this file's directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def save_embeddings(docs_dict):
    """
    Salva os embeddings dos documentos e das queries em arquivos .pt.
    Args:
        docs_dict (dict): {doc_id: texto documento}
    """
    doc_ids, doc_texts = dict_to_list(docs_dict)
    # Carregar modelo pré-treinado SBERT (rápido e eficiente)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Computar embeddings em lote (mais eficiente)
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True, )
    print("Document embeddings shape:", doc_embeddings.shape)  # Debugging line
    # Create a dictionary with embeddings and IDs
    data = {
        'doc_ids': doc_ids,
        'embeddings': doc_embeddings.cpu()  # keep as tensor
    }

    with open("../../data/embeddings/doc_embeddings.pt", "wb") as f:
        pickle.dump(data, f)


# driver code

PATH = 'subset_msmarco_train_0.01_99.pkl'
PATH_DATA = '../data/' + PATH
if __name__ == "__main__":
    with open(PATH_DATA, 'rb') as f:
        data = pickle.load(f)

    queries_dict = {qid: query.text for qid, query in data['queries'].items()}
    print(queries_dict)
    print(f'Quantidade de queries: {len(queries_dict)}')

    docs_dict = {did: doc.text for did, doc in data['docs'].items()}
    print(docs_dict)
    print(f'Quantidade de docs: {len(docs_dict)}')

    # Criando um dicionário para armazenar as relações query-documentos
    qrels_dict = {}

    # Iterando sobre os qrels para construir o dicionário
    for qrel in data['qrels']:
        query_id = qrel.query_id
        doc_id = qrel.doc_id
        
        # Se a query já existe no dicionário, adiciona o doc à lista
        if query_id in qrels_dict:
            qrels_dict[query_id].append(doc_id)
        # Se não existe, cria uma nova lista com o doc
        else:
            qrels_dict[query_id] = [doc_id]

    print(qrels_dict)
    print(f'Quantidade de qrels: {len(qrels_dict)}')


    random.seed(42)

    # Split the queries (assuming queries is a dictionary of {query_id: query_object})
    query_ids = list(queries_dict.keys())  # List of query IDs

    # Shuffle query IDs to ensure a random split
    random.shuffle(query_ids)

    # Split into 80% for training, 20% for validation
    split_ratio = 0.8
    test_query_ids = query_ids[int(len(query_ids) * split_ratio):]

    test_queries_dict = {qid: queries_dict[qid] for qid in test_query_ids}

    print(len(queries_dict))
    print(len(test_queries_dict))

    save_embeddings(docs_dict)