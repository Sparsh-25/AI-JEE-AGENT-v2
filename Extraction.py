from pypdf import PdfReader

reader = PdfReader("data/NCERT-Class-12-Physics-Part-2.pdf")
print(len(reader.pages))

# meta = reader.metadata

# print(meta)
# print(meta.author)
# print(meta.creator)

print(reader.pages[56].extract_text())
