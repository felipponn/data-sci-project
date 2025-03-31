import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def dict_to_list(dict_data):
    return list(dict_data.keys()), list(dict_data.values())

def query_timing(search_function, docs_dict, queries_dict, K=10):
    query_ids, query_texts = dict_to_list(queries_dict)
    query_times = []

    for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
        start_time = time.time()
        search_function(docs_dict, {qid: qtext}, K)
        query_times.append(time.time() - start_time)

    return query_times

def plot_query_times(query_times, search_function):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(query_times)), query_times, marker='o')
    plt.xlabel('Query ID')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time per Query for {search_function.__name__}')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()