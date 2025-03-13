import ir_datasets
import random
import pickle
from tqdm import tqdm

def create_subset(dataset_name, sample_percentage, X, seed=None, verbose=False):
    """
    Cria um subconjunto do dataset com base nas queries selecionadas e na porcentagem de queries a serem amostradas.

    dataset_name: nome do dataset a ser carregado
    sample_percentage: porcentagem de queries a serem amostradas
    X: número de documentos não relevantes a serem amostrados para cada query relevante
    seed: semente para o gerador de números aleatórios
    verbose: se True, imprime informações adicionais durante o processo

    Retorna:
    subset_queries_dict: dicionário com as queries selecionadas
    subset_docs: dicionário com os documentos selecionados
    subset_qrels: lista com os qrels selecionados
    """
    if seed is not None:
        random.seed(seed)

    dataset = ir_datasets.load(dataset_name)
    
    # --- 1) Carrega queries e define subconjunto ---
    queries_list = list(tqdm(dataset.queries_iter(), desc="Lendo Queries"))
    total_queries = len(queries_list)
    if verbose:
        print(f"Total de queries: {total_queries}")

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

    subset_queries_dict = {q.query_id: q for q in subset_queries}
    n_queries_sub = len(subset_queries_dict)
    if n_queries_sub == 0:
        if verbose:
            print("Nenhuma query selecionada, retornando subconjunto vazio.")
        return {}, {}, []

    # --- 2) Filtra qrels e coleta doc_ids relevantes ---
    subset_qrels = []
    relevant_doc_ids = set()
    for qrel in tqdm(dataset.qrels_iter(), desc="Filtrando Qrels"):
        if qrel.query_id in subset_queries_dict:
            subset_qrels.append(qrel)
            relevant_doc_ids.add(qrel.doc_id)

    if verbose:
        print(f"Qrels selecionados: {len(subset_qrels)}")
        print(f"Doc IDs relevantes: {len(relevant_doc_ids)}")

    # Total de documentos não relevantes que queremos
    total_random_needed = X * n_queries_sub

    # --- 3) 1 passada sobre docs_iter() usando reservoir sampling ---
    subset_docs = {}  # Aqui vão doc_id -> doc_obj relevantes + amostrados
    reservoir = []    # Armazena temporariamente os documentos não relevantes escolhidos
    num_docs_vistos_nao_rel = 0

    docs_iter = dataset.docs_iter()
    
    for doc in tqdm(docs_iter, desc="Processando Docs"):
        d_id = doc.doc_id
        if d_id in relevant_doc_ids:
            # Doc relevante entra direto no subset
            subset_docs[d_id] = doc
        else:
            # Doc não relevante => aplicar reservoir sampling
            if len(reservoir) < total_random_needed:
                # Ainda não preenchemos o reservatório
                reservoir.append(doc)
            else:
                # Já temos o reservatório cheio, decide se substitui algum
                j = random.randint(0, num_docs_vistos_nao_rel)
                if j < total_random_needed:
                    reservoir[j] = doc
            num_docs_vistos_nao_rel += 1

    # Por fim, adiciona os docs não relevantes ao subset
    for doc in reservoir:
        subset_docs[doc.doc_id] = doc 
    
    if verbose:
        print(f"Total de Queries no subset: {n_queries_sub}")
        print(f"Total de Documentos relevantes (dos qrels): {len(relevant_doc_ids)}")
        print(f"Total de Documentos amostrados (não relevantes): {len(reservoir)}")
        print(f"Total de Documentos no subset: {len(subset_docs)}")
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
    X = 10  
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
