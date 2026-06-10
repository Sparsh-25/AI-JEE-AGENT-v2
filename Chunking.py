from pathlib import Path

overlap = 450
charcters = 4000
chunks = []
for doc in Path("output").glob("*.txt"):
    text = doc.read_text(encoding='utf-8')
    c = 0
    for i in range(0, len(text), charcters - overlap):
        content = text[i : i + charcters]
        data = {
            "source" : doc,
            "content" : content,
            "chunk_number" : c
        }

        chunks.append(data)

        c+=1


sample = chunks[0]

print(sample)