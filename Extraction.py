import pymupdf
from pathlib import Path
import time

data_dir = Path("data")

output_dir = Path("output")

output_dir.mkdir(exist_ok=True)


CONTENT_START = 14   # go find these in the PDF
JUNK_LINES = { 'Access for free at openstax.org' }   # the boilerplate you've seen



for pdf_path in data_dir.glob("*.pdf"):

    print("Processing", pdf_path.name)
    
    start = time.time()

    full_text = []

    doc = pymupdf.open(pdf_path)

    
    for i, page in enumerate(doc):
        if i < CONTENT_START:
            continue  
        text = page.get_text()

        lines = text.split("\n")
        kept = [line for line in lines if line.strip() not in JUNK_LINES]
        text = "\n".join(kept)

        full_text.append(text)

        print(f"Done with page {i+1}/{len(doc)}")
        print(f"Pages left: {len(doc) - i - 1}")
        print(f"Time elapsed: {time.time()-start:.1f}s\n")

    out_file = output_dir / f"{pdf_path.stem}.txt"
    out_file.write_text("\n\n".join(full_text))
    print(f"Saved → {out_file}\n")