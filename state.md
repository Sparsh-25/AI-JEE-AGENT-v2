"THIS MAY BE A HALLUCINATED ANSWER SINCE EITHER NO CHUNK ID OR WRONG CHUNK ID" always present in output query regardless if the chunk contains chunk_id or not. ----->>> Fixed. Cause was the extraction regex in validate_response, not the LLM - chunk_id\s*:?\s*(\d+) matched only 4 of 12 real citation formats (missed plural chunk_ids, markdown **chunk_id**, chunk_id = N, [chunk_id #N], chunk id N, capitalised). Widened it, and added prompt rule 6 requiring a final "CITATIONS: <id>, <id>" line, parsed as a union with the inline regex. Both paths needed - live runs showed the model using a different inline format each time (【chunk_id 165】 vs 【Chunk‑1 | 2121】) and emitting the CITATIONS line only 1 of 2 runs. Each of the 2 answers passed via a different path.

Latency for all processes - New Query, No warmup, fresh process:
Retrieval elapsed in 1.180909209069796 seconds
Reranker elapsed in 7.008264875039458 seconds
API call response time 3.370765083003789 seconds
[TOTAL Chat] elapsed in 11.584592333063483

Above was cold-start, not steady state - first rerank call is 3-5s on either device, a long-running server pays it once.
Reranker was on CPU the whole time - reranker.py never called .to(device), unlike embeddings.py. Moved it, plus inputs.to(device) (model alone raises "Placeholder storage has not been allocated on MPS device").
Measured with both models resident, 6 calls, torch.mps.synchronize() before timing:
- reranker on CPU -> cold 5.31s, warm median 3.95s
- reranker on MPS -> cold 3.43s, warm median 1.46s, 2.72x warm speedup
- scores bit-identical on all 4 test queries, so this is free
Warm query now: retrieval ~0.4-1.0s + rerank ~1.3s + API ~3s

System answer failing on physics query where calculation is required like speed, our retrieval not working (might have to change the prompt for it to know if no chunks simialar to question then look at the question and then judge or/and remove the retrieval threshold) ----->>> Removed a line from prompt now working fine
Regression from that fix: bi-encoder threshold (0.55) is not a reliable off-topic gate on its own.
Anchor tests on reranker scores:
- "how do I bake chocolate chip cookies" -> 0 chunks pass bi-encoder threshold at all (correctly blocked at retrieval)
- "how to cook pasta" -> 5 chunks DO pass bi-encoder threshold (false positive - lexical overlap with thermal physics chunks about cooking pans/spaghetti/ovens), but reranker scores all clearly negative: -1.87 to -6.26
- "what is an electron" -> reranker scores clearly positive: 0.88 to 2.03
- train question's own reranker score is negative but exact value not recorded

With the strict prompt line removed, "how to cook pasta" now gets answered - LLM hallucinates a recipe mixed with thermal calculations from the irrelevant retrieved chunks, instead of declining. Confirmed regression, not yet fixed. ----->>> Fixed. rerank() now takes threshold=0 and drops anything scoring below it, so "how to cook pasta" returns 0 chunks and hits the existing "no relevant material" message before any API call. The scores were already being computed, integration.py was just throwing them away.

Threshold picked from a 12-query sweep, 6 on-topic 6 off-topic:
- on-topic: lowest score 0.57 ("how do acids act as proton donors"), highest 7.19 ("what is entropy"), all 5 chunks > 0 in every case
- off-topic: 5 of 6 blocked at the bi-encoder, only "how to cook pasta" got through - all scores -6.26 to -1.87, 0 chunks > 0
- gap between highest off-topic (-1.87) and lowest on-topic (0.57) is 2.4 wide, 0 sits in the middle of it
- off-topic queries now cost 0 API calls, they never reach the LLM

Guardrail was refusing legitimate chemistry - \bact\s+as\b blocked "acids act as proton donors", "metal act as a reducing agent", "water act as both an acid and a base", and \bdan\b blocked "Dan Shechtman". Narrowed both to the persona form. Note "how do acids act as proton donors" only appears in the sweep above because of this fix.
Also \bignore\s+(all|previous|above|prior)\s+instructions\b allowed exactly one word between verb and "instructions", so "ignore all previous instructions" passed while "ignore previous instructions" was blocked. Now verb + up to 40 chars + instructions. 13/13 attacks blocked, 0/10 legit queries blocked.

Open: /chat has no rate limit and /health returns ok without checking the index is loaded. Parked for later discussion.

Add a router for. -> Maths and calculation via external library instead of LLM , phy/chem, syllabus lookup