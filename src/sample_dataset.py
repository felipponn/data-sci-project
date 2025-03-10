import ir_datasets
import random


def loader(num_queries):
    # Load MSMARCO dataset
    dataset = ir_datasets.load("msmarco-passage-v2/train")

    # Sample queries
    queries = list(dataset.queries_iter())
    sampled_queries = random.sample(queries, num_queries)

    #returnable dict
    data = dict()

    # Get relevant docs
    qrels_dict = {q.query_id: [] for q in sampled_queries}
    for qrel in dataset.qrels_iter():
        if qrel.query_id in qrels_dict:
            qrels_dict[qrel.query_id].append(qrel.doc_id)

    data["qrels"] = qrels_dict

    # Get documents
    docs_dict = {}
    for doc in dataset.docs_iter():
        if doc.doc_id in {doc_id for doc_ids in qrels_dict.values() for doc_id in doc_ids}:
            docs_dict[doc.doc_id] = doc.text

    data["docs"] = docs_dict

    return data


