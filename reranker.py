import torch
from embeddings import retrieve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()

def rerank(query):

    retrieved = retrieve(query)

    if not retrieved:
        return []

    start = time.perf_counter()

    print(f"starting rerank at {start} for query {query}")

    chunks=[0]*len(retrieved)
    for j in range(len(retrieved)):
        chunks[j] = [query, retrieved[j][1]['text']]
    
    with torch.no_grad():
        inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        reranked_chunks = list(zip([s.item() for s in scores], [t[1] for t in retrieved]))

    sorted_ranks = sorted(reranked_chunks, key = lambda pair: pair[0], reverse=True)

    rerank_elapsed  = time.perf_counter() - start
    print(f"Reranker elapsed in {rerank_elapsed} seconds")
    return sorted_ranks[:5]



if __name__ == "__main__":
    query = "what is organic chemistry and carbon mono dioxide?"
    reranked = rerank(query)
    print(reranked)