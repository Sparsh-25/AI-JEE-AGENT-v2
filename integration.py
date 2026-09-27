import time
from reranker import rerank
from guardrails import validate_query, validate_response, validate_answer
from classifier import classify
from llm import call_llm


def response(query):

    start = time.perf_counter()

    refusal = validate_query(query)

    if refusal:
        return 'Not Allowed (Possibly Misuse/Prompt Injection/Banned Words)'

    route = classify(query)

    if route['route'] == 'out_of_scope':
        return "That is outside JEE Physics, Chemistry and Maths, ask me something from the syllabus and I will help."

    chunks = []

    if route['retrieval']:
        chunks = rerank(route['concept'] if route['calculation'] else query)
        print("reranked chunks ", len(chunks))
        print([(round(s, 2), c['chunk_id']) for s, c in chunks])

    grounded_only = route['retrieval'] and not route['calculation'] and not route['fallback']

    if grounded_only and not chunks:
        return "I don't have relevant material in my sources, if something's missing, mail us at -"

    joined, chunk_id = build_context(chunks)

    messages = [
        {
            "role": "system",
            "content": build_prompt(route, chunks)
        },
        {
            "role": "user",
            "content": build_query(query, joined)
        }
    ]
    before_api_call_time = time.perf_counter()
    print(f"starting api call at {before_api_call_time} for query {query}")
    chat_completion = call_llm(messages)
    api_response_elapse = time.perf_counter() - before_api_call_time
    print(f"API call response time {api_response_elapse} seconds")

    if not chat_completion:
        return "Service is temporarily unavailable. Please try again in a moment."

    answer = chat_completion.choices[0].message.content

    print("Prompt:", chat_completion.usage.prompt_tokens)
    print("Completion:", chat_completion.usage.completion_tokens)
    print("Total:", chat_completion.usage.total_tokens)
    print(f"[TOTAL response] elapsed in {time.perf_counter() - start}")

    if route['calculation'] and not validate_answer(answer):
        print("no ANSWER line in reply")

    if not chunks:
        return answer + '\n\nAnswered from general knowledge, not from your textbooks.'

    if not validate_response(answer, chunk_id):
        return answer + ' \n THIS MAY BE A HALLUCINATED ANSWER SINCE EITHER NO CHUNK ID OR WRONG CHUNK ID'

    return answer


def build_context(chunks):

    blocks = []
    chunk_id = []

    for i, pair in enumerate(chunks):
        chunk = pair[1]
        blocks.append(f"Chunk-{i+1} | chunk_id {chunk['chunk_id']} | source {chunk['source']}\n{chunk['text']}")
        chunk_id.append(chunk['chunk_id'])

    return "\n\n".join(blocks), chunk_id


def build_prompt(route, chunks):

    prompt = TUTOR_RULES

    if chunks:
        prompt += RETRIEVAL_RULES
    else:
        prompt += UNGROUNDED_RULES

    if route['calculation']:
        prompt += CALCULATION_RULES

    if chunks and route['calculation']:
        prompt += MIXED_RULES

    return prompt


def build_query(query, joined):

    if not joined:
        return f"User Query: {query}"

    return f"The chunks are formed as (chunk-number | chunk_id |source | chunk text) and user Query will be define by 'User Query. Context:\n\n{joined}\n\nUser Query: {query}"


TUTOR_RULES = (
    "You are a JEE tutor with excellence in JEE syllabus and materials, your role is not just give answers "
    "but make students undestand about the topics deeply 1) you may use real world analogies "
    "2) Break Complex topcis into manageable steps 3) Encourage critical thinking and problem solving ability "
    "regarding JEE topics 4) Always ask a follow up question for making them understand deeply and to test if "
    "they understood."
)

RETRIEVAL_RULES = (
    " Answer only from the context provided. ALWAYS cite the source and chunk_id used in chunks for answering "
    "the user query. END your reply with a final line in exactly this format: CITATIONS: <chunk_id>, <chunk_id> "
    "- listing the chunk_id value of every chunk you used, never the chunk number."
)

UNGROUNDED_RULES = (
    " You have no textbook extracts for this query, so answer from your own knowledge and say plainly when you "
    "are unsure. Do not invent a citation."
)

CALCULATION_RULES = (
    " Show every step of the working, and state the relation you are using before you substitute any values. "
    "After the full explanation, END your reply with a final line in exactly this format: ANSWER: <result> - "
    "whatever form the result takes, a number with units, an expression, a set or an interval."
)

MIXED_RULES = (
    " The chunks give you the relations this question is built on, they do not contain a worked solution for it. "
    "Use them for the relations and work out the rest yourself. Both final lines come after the full explanation, "
    "the CITATIONS line second to last and the ANSWER line last."
)


if __name__ == '__main__':
    print(response('what is electron'))
