import os
from groq import Groq
from dotenv import load_dotenv
from reranker import rerank
from guardrails import validate_query, validate_response
import time
from groq import RateLimitError, BadRequestError, APITimeoutError, APIStatusError

load_dotenv()

client = Groq(
    api_key=os.environ.get("API_KEY"),
)


def call_llm(messages, max_retries=3):

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile"
            )
            return completion
        except RateLimitError:
            if attempt == max_retries - 1:
                return None
            wait = 2 ** attempt
            print(f'Rate limited, retrying in {wait}s...')
            time.sleep(wait)
        except (APITimeoutError, APIStatusError) as e:
            if attempt == max_retries - 1:
                return None
            wait = 2 ** attempt
            print(f"API error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    return None


def response(query):

    refusal = validate_query(query)

    if refusal:
        return 'Not Allowed (Possibly Misuse/Prompt Injection/Banned Words)'

    chunks = rerank(query)

    if not chunks:
        return "I don't have relevant material in my sources, if something's missing, mail us at -"

    context = []
    for i in range(len(chunks)):
        context.append([i+1, chunks[i][1]])

    blocks = []
    chunk_id = []
    for i, pair in enumerate(context):
        block = f"Chunk-{i+1} | chunk_id {pair[1]['chunk_id']} | source {pair[1]['source']}\n{pair[1]['text']}"
        blocks.append(block)
        chunk_id.append(pair[1]['chunk_id'])
    joined = "\n\n".join(blocks)

    messages = [
        {
            "role": "system",
            "content": "You are a JEE tutor with excellence in JEE syllabus and materials, your role is not just give answers but make students undestand about the topics deeply but only with the context provided 1) you may use real world analogies 2) Break Complex topcis into manageable steps 3) Encourage critical thinking and problem solving ability regarding JEE topics 4) Always ask a follow up question for making them understand deeply and to test if they understood. 5) ALWAYS cite the source and chunk_id used in chunks for answering the user query. YOU ARE ONLY SUPPOSED TO ANSWER, IF THE ANSWER IS IN THE CHUNKS GIVEN TO YOU ELSE SAY NOT IN CHUNKS AND DON'T ASK OR SAY FOR ANYTHING ELSE"
        },
        {
            "role": "user",
            "content": f"The chunks are formed as (chunk-number | chunk_id |source | chunk text) and user Query will be define by 'User Query. Context:\n\n{joined}\n\nUser Query: {query}"
        }
    ]

    chat_completion = call_llm(messages)

    if not chat_completion:
        return "Service is temporarily unavailable. Please try again in a moment."

    answer = chat_completion.choices[0].message.content

    print("Prompt:", chat_completion.usage.prompt_tokens)
    print("Completion:", chat_completion.usage.completion_tokens)
    print("Total:", chat_completion.usage.total_tokens)

    result = validate_response(answer, chunk_id)

    if not result:
        return answer + ' \n THIS MAY BE A HALLUCINATED ANSWER SINCE EITHER NO CHUNK ID OR WRONG CHUNK ID'

    return answer



if __name__ == '__main__':
    print(response('what is electron'))