from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, asdict
from pathlib import Path
import json

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


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
    )

output_dir = Path("chunks")
output_dir.mkdir(exist_ok=True)
docs = Path("output")

for doc in docs.glob("*.txt"):

    text = doc.read_text()
    chunk = text_splitter.create_documents([text])

    for i in range(len(chunk)):
        chunk[i] = Chunks(i, chunk[i].page_content, doc.stem)
    
    write_chunks(chunk, output_dir / f"{doc.stem}.jsonl")

    print("done")