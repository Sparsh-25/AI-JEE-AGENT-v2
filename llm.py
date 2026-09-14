import os
import time
from groq import Groq
from dotenv import load_dotenv
from groq import RateLimitError, BadRequestError, AuthenticationError, NotFoundError, APITimeoutError, APIConnectionError, APIStatusError

load_dotenv()

client = Groq(
    api_key=os.environ.get("API_KEY"),
    max_retries=0,
    timeout=30.0,
)


def call_llm(messages, max_retries=3, **kwargs):

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-120b",
                **kwargs
            )
            return completion
        except RateLimitError:
            if attempt == max_retries - 1:
                return None
            wait = 2 ** attempt
            print(f'Rate limited, retrying in {wait}s...')
            time.sleep(wait)
        except (BadRequestError, AuthenticationError, NotFoundError) as e:
            print(f"Non-retryable API error: {e}")
            return None
        except (APIConnectionError, APIStatusError) as e:
            if attempt == max_retries - 1:
                return None
            wait = 2 ** attempt
            print(f"API error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    return None
