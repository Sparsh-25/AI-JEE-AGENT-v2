from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

@dataclass
class Chunks:
    chunk_id : int
    text: str
    source: str   

def write_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

def read_chunks(path):
    return [Chunks(**json.loads(line)) for line in open(path, encoding="utf-8")]


text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=480,
    chunk_overlap=50
    )

output_dir = Path("chunks")
output_dir.mkdir(exist_ok=True)
docs = Path("output")

for doc in docs.glob("*.txt"):

    text = doc.read_text()
    chunk = text_splitter.create_documents([text])

    kept = []
    cid=0

    for i in range(len(chunk)):
        if len(chunk[i].page_content.strip()) < 20:
            continue

        kept.append(Chunks(cid, chunk[i].page_content, doc.stem))
        cid +=1
    
    write_chunks(kept, output_dir / f"{doc.stem}.jsonl")

    print("done")