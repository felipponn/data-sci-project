import ir_datasets
import random
from tqdm import tqdm
import pickle

def create_subset(
    dataset_name,
    sample_percentage,  
    seed=None,
    verbose=False
):
    """
    Cria um subconjunto do dataset com:
      - sample_percentage das queries (0 < sample_percentage <= 1)
      - todos os qrels associados a essas queries
      - todos os documentos associados a esses qrels
      - todos os scored_docs associados a essas queries

    Retorna:
      subset_queries_dict (dict): {query_id: Query}
      subset_docs (dict): {doc_id: Document}
      subset_qrels (list): lista de Qrels filtrados
      subset_scored_docs (list): lista de ScoredDocs filtrados
    """
    if seed is not None:
        random.seed(seed)
    
    dataset = ir_datasets.load(dataset_name)
    
    # 1) Carrega todas as queries (conseguimos saber o tamanho e usar tqdm com total)
    queries_list = list(tqdm(dataset.queries_iter(), desc="Lendo Queries"))
    total_queries = len(queries_list)
    if verbose:
        print(f"Total de queries: {total_queries}")

    # Ajusta casos extremos de sample_percentage
    if sample_percentage <= 0:
        # Se 0 ou negativo, retorna tudo vazio
        if verbose:
            print("sample_percentage <= 0; retornando subconjunto vazio.")
        return {}, {}, [], []
    elif sample_percentage >= 1:
        # Se >= 1, usamos todas as queries
        subset_queries = queries_list
        if verbose:
            print("sample_percentage >= 1; usando todas as queries.")
    else:
        # Caso contrário, amostramos
        num_to_sample = int(total_queries * sample_percentage)
        num_to_sample = max(num_to_sample, 1)  # Garante ao menos 1
        if verbose:
            print(f"Número de queries a serem selecionadas: {num_to_sample}")
        subset_queries = random.sample(queries_list, num_to_sample)
    
    # Monta dict para checagem rápida de pertinência (query_id -> Query)
    subset_queries_dict = {q.query_id: q for q in subset_queries}
    if not subset_queries_dict:
        # Se não selecionamos nada, retornamos vazio
        if verbose:
            print("Nenhuma query selecionada, retornando subconjunto vazio.")
        return {}, {}, [], []
    
    # 2) Filtra Qrels das queries selecionadas e coleta doc_ids relevantes
    subset_qrels = []
    relevant_doc_ids = set()
    # Aqui não sabemos quantos qrels existem no total, então não passamos "total" para tqdm
    for qrel in tqdm(dataset.qrels_iter(), desc="Filtrando Qrels"):
        if qrel.query_id in subset_queries_dict:
            subset_qrels.append(qrel)
            relevant_doc_ids.add(qrel.doc_id)
    
    if verbose:
        print(f"Qrels selecionados: {len(subset_qrels)}")
        print(f"Doc IDs relevantes: {len(relevant_doc_ids)}")
    
    # 3) Filtra documentos
    subset_docs = {}
    # c = 0
    if relevant_doc_ids:
        for doc in tqdm(dataset.docs_iter(), desc="Filtrando Docs"):
            if doc.doc_id in relevant_doc_ids:
                subset_docs[doc.doc_id] = doc
            # c += 1
            # if c == 500_000:
            #     break
    
    if verbose:
        print(f"Documentos selecionados: {len(subset_docs)}")
    
    # 4) Filtra scored_docs (apenas pelas queries)
    subset_scored_docs = []
    # c = 0
    for scored_doc in tqdm(dataset.scoreddocs_iter(), desc="Filtrando Scored Docs"):
        if scored_doc.query_id in subset_queries_dict:
            subset_scored_docs.append(scored_doc)
        # c += 1
        # if c == 500_000:
        #     break
    
    if verbose:
        print(f"Scored_docs selecionados: {len(subset_scored_docs)}")
    
    return subset_queries_dict, subset_docs, subset_qrels, subset_scored_docs

def create_and_save_dataset(dataset_name, sample_percentage, output_file, seed=None, verbose=False):
    """
    Cria o subset do dataset usando a função create_subset e salva os dados em um arquivo local.
    """
    subset_queries_dict, subset_docs, subset_qrels, subset_scored_docs = create_subset(
        dataset_name, sample_percentage, seed, verbose
    )
    
    data_to_save = {
        'queries': subset_queries_dict,
        'docs': subset_docs,
        'qrels': subset_qrels,
        'scored_docs': subset_scored_docs
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

# Exemplo de uso:
if __name__ == '__main__':
    dataset_name = "msmarco-passage-v2/train"  
    sample_percentage = 0.1           
    output_file = "../data/subset_msmarco_train.pkl"
    seed = 42
    verbose = True

    # Cria e salva o dataset
    create_and_save_dataset(dataset_name, sample_percentage, output_file, seed, verbose)

    # Lê o dataset salvo
    dataset_loaded = load_dataset(output_file)
    print("Dataset carregado:")
    print("Queries:", len(dataset_loaded['queries']))
    print("Docs:", len(dataset_loaded['docs']))
    print("Qrels:", len(dataset_loaded['qrels']))
    print("Scored Docs:", len(dataset_loaded['scored_docs']))
