import torch
from embeddings import retrieve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()
def rerank(query):

    retrieved = retrieve(query)

    chunks=[0]*len(retrieved)
    for j in range(len(retrieved)):
        chunks[j] = [query, retrieved[j][1]['text']]
    
    with torch.no_grad():
        inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        reranked_chunks = list(zip([s.item() for s in scores], [t[1] for t in retrieved]))

    sorted_ranks = sorted(reranked_chunks, key = lambda pair: pair[0], reverse=True)
    return sorted_ranks

            

query = "what is organic chemistry and carbon mono dioxide?"

reranked = rerank(query)

# print(reranked)