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

def save_embeddings(docs_dict, path="../data/embeddings/doc_embeddings.pt"):
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

    with open(path, "wb") as f:
        pickle.dump(data, f)


# driver code

PATH = 'subset_msmarco_train_0.01_99.pkl'
PATH_DATA = '../data/' + PATH
PATH_DATA_CLEAN = '../data/data_clean/' + PATH
if __name__ == "__main__":
    with open(PATH_DATA, 'rb') as f:
        data = pickle.load(f)

    queries_dict = {qid: query.text for qid, query in data['queries'].items()}
    print(queries_dict)
    print(f'Quantidade de queries: {len(queries_dict)}')

    docs_dict = {did: doc.text for did, doc in data['docs'].items()}
    print(docs_dict)
    print(f'Quantidade de docs: {len(docs_dict)}')

    save_embeddings(docs_dict)
    
    #clean case
    with open(PATH_DATA_CLEAN, 'rb') as f:
        data_clean = pickle.load(f)

    docs_dict_clean = data_clean['docs_dict']
    queries_dict_clean = data_clean['queries_dict']


    print("Queries limpas:")
    print(len(queries_dict_clean))
    print("\nDocs limpos:")
    print(len(docs_dict_clean))
    
    
    save_embeddings(docs_dict_clean, path="../data/embeddings/doc_embeddings_clean.pt")