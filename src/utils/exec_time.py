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