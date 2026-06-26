import os
from groq import Groq
from dotenv import load_dotenv
from reranker import rerank

load_dotenv()

client = Groq(
    api_key=os.environ.get("API_KEY"),
)

query = "what is organic chem"

chunks = rerank(query)

context = []
for i in range(len(chunks)):

    context.append([i+1, chunks[i][1]])



blocks = []
for i, pair in enumerate(context):
    block = f"Chunk-{i+1} | chunk_id {pair[1]['chunk_id']} | source {pair[1]['source']}\n{pair[1]['text']}"
    blocks.append(block)
joined = "\n\n".join(blocks)

def response(query):

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content" : "You are a JEE tutor with excellence in JEE syllabus and materials, your role is not just give answers but make students undestand about the topics deeply but only with the context provided 1) you may use real world analogies 2) Break Complex topcis into manageable steps 3) Encourage critical thinking and problem solving ability regarding JEE topics 4) Always ask a follow up question for making them understand deeply and to test if they understood. 5) ALWAYS cite the source and chunk_id used in chunks for answering the user query. YOU ARE ONLY SUPPOSED TO ANSWER, IF THE ANSWER IS IN THE CHUNKS GIVEN TO YOU ELSE SAY NOT IN CHUNKS AND DON'T ASK OR SAY FOR ANYTHING ELSE"
            }, 
            {
                "role": "user",
                "content": f"The chunks are formed as (chunk-number | chunk_id |source | chunk text) and user Query will be define by 'User Query. Context:\n\n{joined}\n\nUser Query: {query}"
            }
        ],
        model="llama-3.3-70b-versatile"
    )




# print(context)