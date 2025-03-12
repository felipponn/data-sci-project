import ir_datasets
import random
import pickle
from tqdm import tqdm

def create_subset(
    dataset_name,
    sample_percentage,
    X,                 # número de documentos aleatórios (falsos) a adicionar (globalmente)
    seed=None,
    verbose=False
):
    """
    Cria um subconjunto do dataset com:
      - 'sample_percentage' das queries (0 < sample_percentage <= 1)
      - todos os qrels associados a essas queries
      - para cada query selecionada, pega todos os documentos relevantes
      - adicionalmente, sorteamos X * (número de queries selecionadas) documentos
        (do total do dataset, excluindo os relevantes) para compor o conjunto final
      - não altera em nada os qrels (ou seja, não insere qrels com relevância=0)
    
    Retorna:
      - subset_queries_dict (dict): {query_id: Query}
      - subset_docs (dict): {doc_id: Document}
      - subset_qrels (list): lista de qrels filtrada (apenas para as queries selecionadas)
    """

    if seed is not None:
        random.seed(seed)
    
    dataset = ir_datasets.load(dataset_name)
    
    # 1) Carrega todas as queries
    queries_list = list(tqdm(dataset.queries_iter(), desc="Lendo Queries"))
    total_queries = len(queries_list)
    if verbose:
        print(f"Total de queries: {total_queries}")

    # 2) Seleciona um subconjunto de queries com base em sample_percentage
    if sample_percentage <= 0:
        if verbose:
            print("sample_percentage <= 0; retornando subconjunto vazio.")
        return {}, {}, []
    elif sample_percentage >= 1:
        # Pega todas as queries
        subset_queries = queries_list
        if verbose:
            print("sample_percentage >= 1; usando todas as queries.")
    else:
        num_to_sample = int(total_queries * sample_percentage)
        num_to_sample = max(num_to_sample, 1)
        if verbose:
            print(f"Número de queries a serem selecionadas: {num_to_sample}")
        subset_queries = random.sample(queries_list, num_to_sample)

    # Dicionário {query_id: objeto da Query}
    subset_queries_dict = {q.query_id: q for q in subset_queries}
    n_queries_sub = len(subset_queries_dict)
    if n_queries_sub == 0:
        if verbose:
            print("Nenhuma query selecionada, retornando subconjunto vazio.")
        return {}, {}, []

    # 3) Filtra qrels: pega somente aqueles que pertencem às queries selecionadas
    #    e coleta o conjunto de doc_ids relevantes.
    subset_qrels = []
    relevant_doc_ids = set()
    for qrel in tqdm(dataset.qrels_iter(), desc="Filtrando Qrels"):
        if qrel.query_id in subset_queries_dict:
            subset_qrels.append(qrel)
            relevant_doc_ids.add(qrel.doc_id)

    if verbose:
        print(f"Qrels selecionados: {len(subset_qrels)}")
        print(f"Doc IDs relevantes (dos qrels): {len(relevant_doc_ids)}")

    # 4) Lê todos os documentos apenas uma vez, guardando em estruturas:
    #    - doc_dict: {doc_id: doc_obj}
    #    - all_doc_ids: lista (ou set) de todos os doc_ids
    doc_dict = {}
    all_doc_ids = []
    for doc in tqdm(dataset.docs_iter(), desc="Lendo todos os Docs (1a e única vez)"):
        doc_dict[doc.doc_id] = doc
        all_doc_ids.append(doc.doc_id)

    all_doc_ids = set(all_doc_ids)

    # 5) Precisamos sortear X * n_queries_sub documentos "aleatórios"
    #    que não estejam em 'relevant_doc_ids'. Assim não poluímos os qrels originais.
    non_relevant_doc_ids = list(all_doc_ids - relevant_doc_ids)
    
    # Se o dataset tiver menos docs que o necessário, random.sample gera erro.
    # Por isso, pegamos o mínimo.
    total_random_needed = X * n_queries_sub
    total_random_needed = min(total_random_needed, len(non_relevant_doc_ids))

    random_docs_sample = random.sample(non_relevant_doc_ids, total_random_needed)
    
    # 6) Conjunto final de doc_ids = relevantes (dos qrels) + amostra aleatória
    final_doc_ids = relevant_doc_ids.union(random_docs_sample)

    # 7) Monta o dicionário final de documentos (subset_docs)
    subset_docs = {doc_id: doc_dict[doc_id] for doc_id in final_doc_ids}

    if verbose:
        print(f"Total de queries no subset: {n_queries_sub}")
        print(f"Total de documentos no subset: {len(subset_docs)}")
        print(f"Total de Qrels (original, sem alterações): {len(subset_qrels)}")

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
    X = 10  # Número de documentos aleatórios (falsos) a serem adicionados, para todo o conjunto
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
