import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import semaphore
from logger_setup import logger
from utils import call_chat_completions, extract_json_string, sanitize_json_string


async def generate_description(prompt, product_title, description_type):
    assistant_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = None
    logger.info(f"Generating new {description_type} for the product '{product_title}'.")
    loop = asyncio.get_event_loop()

    async with semaphore:
        try:
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    call_chat_completions,
                    "gpt-4",
                    assistant_messages,
                )
            if response and response.get("choices") and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]

                # Extract the JSON object string from the generated text
                json_string = extract_json_string(generated_text)

                # Sanitize the JSON string and parse it as a JSON object
                data = sanitize_json_string(json_string)

                return data
        except Exception as error:
            logger.error(f"Error during OpenAI API call: {error}")
            return None
