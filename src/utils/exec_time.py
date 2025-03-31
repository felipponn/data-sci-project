import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def dict_to_list(dict_data):
    """
    Converte um dicionário em duas listas: uma com as chaves e outra com os valores.
    
    Args:
        dict_data (dict): dicionário a ser convertido
        
    Returns:
        tuple: duas listas, uma com as chaves e outra com os valores do dicionário
    """
    return list(dict_data.keys()), list(dict_data.values())

def query_timing(search_function, docs_dict, queries_dict, K=10):
    """
    Calcula o tempo de processamento para cada query individualmente.

    Args:
        search_function: função de busca (random_search, tfidf_search, bm25_search, etc.)
        docs_dict: dicionário com {doc_id: texto do documento}
        queries_dict: dicionário com {query_id: texto da query}
        K: número de documentos a retornar por query (default=10)

    Returns:
        list: tempos de processamento para cada query
    """
    query_ids, query_texts = dict_to_list(queries_dict)
    query_times = []

    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        start_time = time.time()
        search_function(docs_dict, {qid: qtext}, K)
        query_times.append(time.time() - start_time)

    return query_times

def plot_query_times(query_times, search_function):
    """
    Plota os tempos de execução para cada query.
    
    Args:
        query_times (list): lista de tempos de execução para cada query
        search_function: função de busca utilizada (para título do gráfico)
    """
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(query_times)), query_times, marker='o')
    plt.xlabel('Query ID')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time per Query for {search_function.__name__}')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()