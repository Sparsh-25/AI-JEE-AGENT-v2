from pathlib import Path

def chunking(text, seperator=["\n\n", "\n", " ", ""], chunk_size =512 ):

    if len(text) <= chunk_size:
        return [text]
    if not seperator or seperator[0] == "":
        forced=[]
        for i in range(0, len(text), chunk_size):
            chunk  = text[i:i+chunk_size]
            if len(chunk) > 50:
                forced.append(chunk)

        return forced

    else:
        current_step = seperator[0]
        remaining_steps = seperator[1:]

        split = text.split(current_step)
        chunks = []

        for piece in split:

            if not piece.strip():
                continue

            if len(piece) <= chunk_size:
                chunks.append(piece)
            else:
                chunk = chunking(piece, remaining_steps, chunk_size)
                chunks.extend(chunk)


        return chunks

output_dir = Path("Chunk")

docs = Path("output")

for doc in docs.glob("*.txt"):

    text = doc.read_text()
    chunk = chunking(text)
    print("done")
