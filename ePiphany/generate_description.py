import asyncio
import codecs
import json
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import openai
from config import semaphore
from logger_setup import logger

from database import c


def call_chat_completions(model, messages):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    print(f"Raw API Response: {response}")
    return response


def extract_json_string(generated_text):
    # Find the start and end of the JSON object
    start = generated_text.find("{")
    end = generated_text.rfind("}")

    # Extract the JSON object
    json_string = generated_text[start : end + 1]

    return json_string


def sanitize_json_string(json_string):
    try:
        # Try to load the JSON string. If it's valid JSON and there are no control characters, this will succeed.
        data = json.loads(json_string)
    except json.JSONDecodeError:
        # If there's an error, remove control characters and unwanted backslashes from the string and try again.
        sanitized_string = remove_control_characters_and_unwanted_backslashes(
            json_string
        )
        data = json.loads(sanitized_string)

    return data


def remove_control_characters_and_unwanted_backslashes(s):
    sanitized_string = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    sanitized_string = codecs.decode(sanitized_string.encode().decode("unicode_escape"))

    return sanitized_string


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


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
                if isinstance(response, bytes):
                    response = response.decode("utf-8")
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
