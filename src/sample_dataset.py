import ir_datasets
import random

# Load MSMARCO dataset
dataset = ir_datasets.load("msmarco-passage/train")

# Define subset size
num_queries = 1000

# Sample queries
queries = list(dataset.queries_iter())
sampled_queries = random.sample(queries, num_queries)

# Get relevant docs
qrels_dict = {q.query_id: [] for q in sampled_queries}
for qrel in dataset.qrels_iter():
    if qrel.query_id in qrels_dict:
        qrels_dict[qrel.query_id].append(qrel.doc_id)

# Get documents
docs_dict = {}
for doc in dataset.docs_iter():
    if doc.doc_id in {doc_id for doc_ids in qrels_dict.values() for doc_id in doc_ids}:
        docs_dict[doc.doc_id] = doc.text

# Save the smaller dataset
with open("small_msmarco.txt", "w") as f:
    for q in sampled_queries:
        f.write(f"Query: {q.text}\n")
        for doc_id in qrels_dict[q.query_id]:
            f.write(f"\tDoc: {docs_dict.get(doc_id, 'Missing')}\n")

print("Subset created successfully!")
