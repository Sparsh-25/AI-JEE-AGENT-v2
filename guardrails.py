import re 

def validate_query(query):

    if not query or not query.strip():

        return 'refusal'

    if len(query)> 600:

        return 'refusal'
    
    result = check_input(query)

    if not result:
        return 'refusal'
    
    return

def check_input(text: str):

    if PROMPT_INJECTION_RE.search(text):
        return False

    if MISUSE_RE.search(text):
        return False

    if BAD_WORD_RE.search(text):
        return False

    return True

def validate_response(answer, chunk_id):

    cited = set(int(x) for x in re.findall(r'chunk[_\s-]?ids?\b[\s\S]{0,12}?(\d+)', answer, re.IGNORECASE))

    line = re.search(r'CITATIONS\s*:\s*([\d,\s]+)', answer, re.IGNORECASE)
    if line:
        cited |= set(int(x) for x in re.findall(r'\d+', line.group(1)))

    if not cited:
        return False
    
    if not cited & set(chunk_id):
        return False
    
    return True
    
PROMPT_INJECTION_PATTERNS = [
    r"\b(ignore|forget|disregard)\b[\s\S]{0,40}?\binstructions\b",
    r"\boverride\s+(the\s+)?system\b",
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\breveal\s+(your\s+)?prompt\b",
    r"\bshow\s+(your\s+)?system\s+prompt\b",
    r"\bprint\s+(your\s+)?instructions\b",
    r"\bhidden\s+prompt\b",
    r"\byou\s+(should\s+|must\s+|will\s+|can\s+|to\s+)?act\s+as\b",
    r"\bact\s+as\s+(a\s+|an\s+)?(dan|ai|assistant|chatbot|jailbroken|unrestricted|uncensored)\b",
    r"\bpretend\s+to\s+be\b",
    r"\byou\s+are\s+now\b",
    r"\bdo\s+anything\s+now\b",
    r"\bjailbreak\b",
]

# Attempts to use the model outside JEE
MISUSE_PATTERNS = [
    r"\bwrite\s+(a\s+)?resume\b",
    r"\bwrite\s+(an\s+)?email\b",
    r"\bsolve\s+leetcode\b",
    r"\bwrite\s+code\b",
    r"\bgenerate\s+sql\b",
    r"\bhack\b",
    r"\bpassword\b",
    r"\bcredit\s+card\b",
    r"\bbitcoin\b",
    r"\bstock\s+prediction\b",
    r"\bmedical\s+advice\b",
    r"\blegal\s+advice\b",
]

# Offensive words (small example)
BAD_WORD_PATTERNS = [
    r"\bfuck\b",
    r"\bshit\b",
    r"\bbitch\b",
    r"\basshole\b",
    r"\bbastard\b",
    r"\bdick\b",
    r"\bpussy\b",
]

PROMPT_INJECTION_RE = re.compile(
    "|".join(PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE,
)

MISUSE_RE = re.compile(
    "|".join(MISUSE_PATTERNS),
    re.IGNORECASE,
)

BAD_WORD_RE = re.compile(
    "|".join(BAD_WORD_PATTERNS),
    re.IGNORECASE,
)
