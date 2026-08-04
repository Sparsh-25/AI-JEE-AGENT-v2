from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, asdict
import json
import re

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

@dataclass
class Chunks:
    chunk_id : int
    text: str
    source: str
    page: int

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

cid=0
for doc in docs.glob("*.txt"):

    text = doc.read_text()
    segments = re.split(r'<<<PAGE (\d+)>>>', text) # The list alternates: number, text, number, text, number, text.

    kept = []

    for j in range(1, len(segments), 2): # two steps because alternate format and from 1 because junk/whitespaces
        page_num = int(segments[j])
        page_text = segments[j + 1]

        page_chunks = text_splitter.create_documents([page_text])

        for c in page_chunks:
            if len(c.page_content.strip()) < 20:
                continue
            kept.append(Chunks(cid, c.page_content, doc.stem, page_num))
            cid += 1

    write_chunks(kept, output_dir / f"{doc.stem}.jsonl")

    print(f"done {doc.stem}: {len(kept)} chunks")