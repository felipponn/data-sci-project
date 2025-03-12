import ir_datasets
import random
import pickle
from collections import defaultdict
from tqdm import tqdm
from ir_datasets.formats import GenericQrel

def create_subset(
    dataset_name,
    sample_percentage,
    X,  # NOVO: número de documentos aleatórios (falsos positivos) a adicionar
    seed=None,
    verbose=False
):
    """
    Cria um subconjunto do dataset com:
      - sample_percentage das queries (0 < sample_percentage <= 1)
      - para cada query selecionada, pega todos os qrels relevantes
      - adiciona X documentos aleatórios (falsos positivos) para cada query
      - retorna:
         subset_queries_dict (dict): {query_id: Query}
         subset_docs (dict): {doc_id: Document}
         subset_qrels (list): lista de Qrels (GenericQrel) com relevância 1 para relevantes e 0 para falsos positivos
    """
    if seed is not None:
        random.seed(seed)
    
    dataset = ir_datasets.load(dataset_name)
    
    # 1) Carrega todas as queries
    queries_list = list(tqdm(dataset.queries_iter(), desc="Lendo Queries"))
    total_queries = len(queries_list)
    if verbose:
        print(f"Total de queries: {total_queries}")

    # Ajusta sample_percentage
    if sample_percentage <= 0:
        if verbose:
            print("sample_percentage <= 0; retornando subconjunto vazio.")
        return {}, {}, []
    elif sample_percentage >= 1:
        subset_queries = queries_list
        if verbose:
            print("sample_percentage >= 1; usando todas as queries.")
    else:
        num_to_sample = int(total_queries * sample_percentage)
        num_to_sample = max(num_to_sample, 1)
        if verbose:
            print(f"Número de queries a serem selecionadas: {num_to_sample}")
        subset_queries = random.sample(queries_list, num_to_sample)
    
    # Dicionário {query_id: Query}
    subset_queries_dict = {q.query_id: q for q in subset_queries}
    if not subset_queries_dict:
        if verbose:
            print("Nenhuma query selecionada, retornando subconjunto vazio.")
        return {}, {}, []

    # 2) Carrega todos os qrels e associa apenas às queries selecionadas
    query_to_rel_docs = defaultdict(set)
    for qrel in tqdm(dataset.qrels_iter(), desc="Carregando Qrels"):
        if qrel.query_id in subset_queries_dict:
            query_to_rel_docs[qrel.query_id].add(qrel.doc_id)
    
    # 3) Precisamos do conjunto de todos os doc_ids do dataset
    #    para sortear os falsos positivos (X docs)
    all_doc_ids = []
    for doc in tqdm(dataset.docs_iter(), desc="Lendo todos os Doc IDs"):
        all_doc_ids.append(doc.doc_id)
    all_doc_ids = set(all_doc_ids)

    # 4) Agora vamos construir os qrels finais e a lista final de doc_ids
    #    Para cada query -> doc_ids relevantes + X negativos aleatórios
    subset_qrels = []   # lista de GenericQrel
    final_doc_ids = set()

    for q_id in subset_queries_dict:
        # Relevantes para essa query
        rel_docs = query_to_rel_docs[q_id]
        
        # Sorteia X documentos que não são relevantes
        non_rel = list(all_doc_ids - rel_docs)
        if len(non_rel) >= X:
            neg_docs = random.sample(non_rel, X)
        else:
            # Caso (raro) em que não há docs suficientes fora dos relevantes
            neg_docs = non_rel
        
        # Adiciona no Qrel: relevância = 1 para relevantes, 0 para negativos
        # (iteration=0, pois não estamos usando esse campo no momento)
        for d_id in rel_docs:
            subset_qrels.append(GenericQrel(q_id, d_id, 1, 0))
        for d_id in neg_docs:
            subset_qrels.append(GenericQrel(q_id, d_id, 0, 0))
        
        # Unifica doc_ids finais para essa query
        final_doc_ids_for_q = rel_docs.union(neg_docs)
        final_doc_ids.update(final_doc_ids_for_q)
    
    # 5) Filtra e carrega no dicionário apenas os documentos usados (relevantes + falsos positivos)
    #    Precisamos iterar de novo pelo docs_iter() para obter o texto e metadados.
    subset_docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Filtrando Docs finais"):
        if doc.doc_id in final_doc_ids:
            subset_docs[doc.doc_id] = doc

    if verbose:
        print(f"Total de queries no subset: {len(subset_queries_dict)}")
        print(f"Total de documentos no subset: {len(subset_docs)}")
        print(f"Total de Qrels (rel + falsos positivos): {len(subset_qrels)}")
    
    return subset_queries_dict, subset_docs, subset_qrels

def create_and_save_dataset(dataset_name, sample_percentage, X, output_file, seed=None, verbose=False):
    """
    Cria o subset do dataset usando a função create_subset e salva os dados em um arquivo local.
    """
    subset_queries_dict, subset_docs, subset_qrels = create_subset(
        dataset_name, sample_percentage, X, seed, verbose
    )
    
    data_to_save = {
        'queries': subset_queries_dict,
        'docs': subset_docs,
        'qrels': subset_qrels
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    if verbose:
        print(f"Dataset salvo com sucesso em '{output_file}'.")

def load_dataset(input_file):
    """
    Lê e retorna os dados do dataset salvos localmente a partir do arquivo input_file.
    """
    with open(input_file, 'rb') as f:
        data_loaded = pickle.load(f)
    return data_loaded


# Exemplo de uso
if __name__ == '__main__':
    dataset_name = "msmarco-passage-v2/train"
    sample_percentage = 0.05
    X = 10  # Número de documentos falsos positivos a serem adicionados por query
    output_file = "../data/subset_msmarco_train.pkl"
    seed = 42
    verbose = True

    # Cria e salva o dataset
    create_and_save_dataset(dataset_name, sample_percentage, X, output_file, seed, verbose)

    # Lê o dataset salvo
    dataset_loaded = load_dataset(output_file)
    print("\nDataset carregado:")
    print("Queries:", len(dataset_loaded['queries']))
    print("Docs:", len(dataset_loaded['docs']))
    print("Qrels:", len(dataset_loaded['qrels']))
