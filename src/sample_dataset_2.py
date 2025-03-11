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
      - todos os scored_docs associados a essas queries
      - todos os documentos associados aos qrels e scored_docs

    Retorna:
      subset_queries_dict (dict): {query_id: Query}
      subset_docs (dict): {doc_id: Document}
      subset_qrels (list): lista de Qrels filtrados
      subset_scored_docs (list): lista de ScoredDocs filtrados
    """
    if seed is not None:
        random.seed(seed)
    
    dataset = ir_datasets.load(dataset_name)
    
    # 1) Carrega todas as queries (usando tqdm para acompanhar o progresso)
    queries_list = list(tqdm(dataset.queries_iter(), desc="Lendo Queries"))
    total_queries = len(queries_list)
    if verbose:
        print(f"Total de queries: {total_queries}")

    # Ajusta sample_percentage
    if sample_percentage <= 0:
        if verbose:
            print("sample_percentage <= 0; retornando subconjunto vazio.")
        return {}, {}, [], []
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
    
    # Cria dicionário para acesso rápido
    subset_queries_dict = {q.query_id: q for q in subset_queries}
    if not subset_queries_dict:
        if verbose:
            print("Nenhuma query selecionada, retornando subconjunto vazio.")
        return {}, {}, [], []
    
    # 2) Filtra Qrels e coleta doc_ids relevantes (dos qrels)
    subset_qrels = []
    relevant_doc_ids = set()
    for qrel in tqdm(dataset.qrels_iter(), desc="Filtrando Qrels"):
        if qrel.query_id in subset_queries_dict:
            subset_qrels.append(qrel)
            relevant_doc_ids.add(qrel.doc_id)
    
    if verbose:
        print(f"Qrels selecionados: {len(subset_qrels)}")
        print(f"Doc IDs relevantes (dos qrels): {len(relevant_doc_ids)}")
    
    # 3) Filtra scored_docs antes dos documentos
    subset_scored_docs = []
    for scored_doc in tqdm(dataset.scoreddocs_iter(), desc="Filtrando Scored Docs"):
        if scored_doc.query_id in subset_queries_dict:
            subset_scored_docs.append(scored_doc)
            # Adiciona o doc_id dos scored_docs ao conjunto de relevantes
            relevant_doc_ids.add(scored_doc.doc_id)
    
    if verbose:
        print(f"Scored_docs selecionados: {len(subset_scored_docs)}")
        print(f"Doc IDs relevantes atualizados (qrels + scored_docs): {len(relevant_doc_ids)}")
    
    # 4) Filtra documentos utilizando os doc_ids coletados
    subset_docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Filtrando Docs"):
        if doc.doc_id in relevant_doc_ids:
            subset_docs[doc.doc_id] = doc

    if verbose:
        print(f"Documentos selecionados: {len(subset_docs)}")
    
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
    sample_percentage = 0.01           
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
