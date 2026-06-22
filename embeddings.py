from transformers import AutoModel, AutoTokenizer
import numpy
import torch
from pathlib import Path
import json
import numpy as np

if torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
model.to(device)
model.eval()

def encode(text, mode='chunk'): 

    L =[]
    
    if mode == 'query':
        
        for i in range(len(text)):
            L.append(f"Represent this sentence for searching relevant passages: {text[i]}")

    else:
        L = text

    encoded_input = tokenizer(L, padding = True, truncation=True, return_tensors='pt')

    encoded_input = encoded_input.to(device)


    with torch.no_grad():
        model_output = model(**encoded_input)

        embedding = model_output[0][ :, 0]

    embedding = torch.nn.functional.normalize(embedding, p =2, dim=1)

    return embedding


def retrieve(query, k=5, threshold=0.6):

    query_vec = encode([query], mode = 'query')[0].cpu().numpy()

    score = embedded @ query_vec

    top = np.argpartition(-score, k)[:k]      # the k highest-scoring indices (unordered)
    top = top[np.argsort(-score[top])]        # sort those k by score, descending
    return [(float(score[i]), chunks[i]) for i in top ]  #if score[i] >= threshold to put


def encode_all():

    matrix = []
    all_chunks = []
    for doc in input_dir.glob("*.jsonl"):

        content = doc.read_text()

        dics = [json.loads(line) for line in content.splitlines() if line.strip()]

        all_chunks.extend(dics)

        for i in range(0, len(dics), 32):

            embed = encode([d['text'] for d in dics[i:i+32]]) 
            matrix.append(embed.cpu().numpy())

            print(f"batch {i//32}")
    
    matrix = np.vstack(matrix)


    np.save(out_dir / "embeddings.npy", matrix)

    with open(out_dir / "chunks.json", "w") as f:
        json.dump(all_chunks, f)  



embedded = np.load('embeddings/embeddings.npy')
chunks = json.load(open("embeddings/chunks.json"))

input_dir = Path('chunks')

out_dir = Path('embeddings')


# print(retrieve("what is Newton's second law")) # 0.7 around
# print(retrieve("how many hydrogen atoms in haloalkene and mehtylamine")) 0.6
# print(retrieve("how do I bake chocolate chip cookies")) # 0.5 around


encode_all()