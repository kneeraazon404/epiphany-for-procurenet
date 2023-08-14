# 1. Import the necessary libraries

import asyncio
import json
import logging
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from html import unescape

import aiohttp
import asyncpg
import openai
import simplejson as json
from bs4 import BeautifulSoup
from halo import Halo
from woocommerce import API

# Spinner setup
success_message = "Loading success"
failed_message = "Loading failed"
unicorn_message = "Loading unicorn"

# 2. Use the spinner for the loading effect
spinner = Halo(text=success_message, spinner="dots")
try:
    spinner.start()
    time.sleep(1)
    spinner.succeed()
    spinner.start(failed_message)
    time.sleep(1)
    spinner.fail()
    spinner.start(unicorn_message)
    time.sleep(1)
    spinner.stop_and_persist(symbol="🦄".encode("utf-8"), text=unicorn_message)
except (KeyboardInterrupt, SystemExit):
    spinner.stop()


# 3. Set up API keys and other constants
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

wcapi = API(
    url="https://procure-net.com",
    consumer_key="ck_302221e1ae92c8f9325f4d6de6b1da8aabf5353d",
    consumer_secret="cs_c49195e3967f050a758c6ddddd296b94648750cb",
    version="wc/v3",
)


# 4. Set up the logging configuration
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(log_directory, "logfile.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)


conn = None  # global variable to hold the connection


async def create_db_connection():
    global conn
    conn = await asyncpg.create_pool(
        database="apppgsqldb",
        user="apppgsqluser",
        password="p0$tGrr3$qladM!n18072023",
        host="43.207.8.208",
        port="5432",
        min_size=5,  # Minimum number of connections in the pool
        max_size=20,  # Maximum number of connections in the pool
    )





async def create_tables():
    async with conn.acquire() as connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS updated_products (
                id SERIAL PRIMARY KEY,
                last_update TIMESTAMP WITHOUT TIME ZONE,
                product_name VARCHAR(255),
                product_url VARCHAR(255),
                retry_failed BOOLEAN,
                short_description TEXT,
                long_description TEXT,
                seo_title VARCHAR(255),
                seo_meta_description TEXT,
                focus_keywords TEXT,
                tags VARCHAR(255),
                categories VARCHAR(255)
            );

        CREATE TABLE IF NOT EXISTS failed_products (
            id SERIAL PRIMARY KEY,
            product_name TEXT,
            product_url TEXT,
            short_description TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS generated_responses (
            id SERIAL PRIMARY KEY,
            short_description TEXT,
            long_description TEXT,
            seo_title TEXT,
            seo_meta_description TEXT,
            focus_keywords TEXT,
            tags TEXT,
            categories TEXT[]  -- This defines an array data type
        );
    """
        )

    await create_tables()


logger.info(
    "PostgreSQL database connected and table updated_products created if not exist."
)

# 6. Implement the async functions and logic for product updates, WooCommerce API operations, and error handling
semaphore = asyncio.Semaphore(25)


async def save_to_generated_responses(raw_response, error_message):
    query = "INSERT INTO generated_responses (short_description, long_description) VALUES ($1, $2)"
    try:
        async with conn.acquire() as connection:  # Use the connection pool
            await connection.execute(query, raw_response, error_message)
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error: {str(e)}")


def clean_html(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    cleaned = str(
        soup
    )  # Gets HTML as a plain string without any additional whitespace/newlines
    return cleaned


def prettify_html(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    pretty_html = soup.prettify()  # Gets nicely formatted HTML
    return pretty_html


def extract_json_string(generated_text):
    # Find all occurrences of '{' and '}'
    start_indices = [i for i, char in enumerate(generated_text) if char == "{"]
    end_indices = [i for i, char in enumerate(generated_text) if char == "}"]

    # Ensure there's at least one pair of '{' and '}'
    if start_indices and end_indices:
        # Use the last '{' and the first '}' after it to extract the JSON string
        start = start_indices[-1]
        end = next(idx for idx in end_indices if idx > start)
        json_string = generated_text[start : end + 1]
    else:
        logger.error(
            "Failed to extract JSON string from the generated text due to delimiter issues."
        )
        return "{}"

    return json_string


def sanitize_json_string(json_string):
    # Initially sanitize the json_string
    sanitized_string = remove_control_characters_and_unwanted_backslashes(json_string)

    try:
        # Try to load the sanitized JSON string.
        data = json.loads(sanitized_string)
    except json.JSONDecodeError as error:
        # If there's still an error, log the problematic string and error
        logger.error(
            f"Failed to parse JSON after sanitization. Error: {error}. JSON String: {sanitized_string}"
        )
        return {}

    return data


def remove_control_characters_and_unwanted_backslashes(s):
    """
    Further sanitize the given string by:
    1. Removing specific unwanted control characters, including newlines.
    2. Handling double backslashes and other escape sequences.
    """
    # Removing specific unwanted control characters (e.g. BEL, FF, VT, newline).
    unwanted_chars = ["\a", "\f", "\v", "\n"]
    for char in unwanted_chars:
        s = s.replace(char, "")

    # Handling double backslashes and other escape sequences.
    sanitized_string = ""
    i = 0
    while i < len(s):
        if s[i] == "\\":
            # If double backslashes are found, replace them with a single backslash
            if i + 1 < len(s) and s[i + 1] == "\\":
                sanitized_string += "\\"
                i += 2
            # If a valid escape sequence is found, preserve it
            elif i + 1 < len(s) and s[i + 1] in [
                "n",
                "r",
                "t",
                "b",
                "f",
                '"',
                "'",
                "\\",
            ]:
                sanitized_string += s[i : i + 2]
                i += 2
            else:
                sanitized_string += s[i]
                i += 1
        else:
            sanitized_string += s[i]
            i += 1

    return sanitized_string


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


async def get_categories(session, wcapi):
    categories = []
    page = 1
    per_page = 100  # Maximum limit

    while True:
        async with semaphore, session.get(
            f"{wcapi.url}/wp-json/wc/v3/products/categories",
            params={"page": page, "per_page": per_page},
            auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
            ssl=False,
        ) as resp:
            page_categories = await resp.json()
            categories.extend(page_categories)

            # Break the loop if we have fetched all the categories
            if len(page_categories) < per_page:
                break

            page += 1

    return categories


async def get_category_data(session, wcapi, category_id):
    async with semaphore, session.get(
        f"{wcapi.url}/wp-json/wc/v3/products/categories/{category_id}",
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        ssl=False,
    ) as resp:
        category_data = await resp.json()
        category_name = category_data.get("name")
        product_count = category_data.get("count")
        return category_name, product_count


async def call_gpt4_function(prompt, function_name, functions):
    assistant_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    data = None
    loop = asyncio.get_event_loop()

    logger.info(f"Calling function '{function_name}' with prompt '{prompt}'.")

    async with semaphore:
        try:
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    partial(
                        openai.ChatCompletion.create,
                        model="gpt-4",
                        messages=assistant_messages,
                        function_call={"name": function_name},
                        functions=functions,
                    ),
                )

                print(f"Response: {response}")  # For debugging
                if response is None or not isinstance(response, dict):
                    logger.error("Received None response from OpenAI API. Retrying...")
                    return None
                if response.get("choices") and len(response["choices"]) > 0:
                    generated_text = response["choices"][0]["message"]["content"]

                    # Extract the JSON object string from the generated text
                    json_string = extract_json_string(generated_text)

                    # Sanitize the JSON string and parse it as a JSON object
                    data = sanitize_json_string(json_string)

                    return data

            if response and response.get("choices") and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]

                if "function_call" in message:
                    # The generated text is inside the "arguments" key in the "function_call" object
                    generated_text = message["function_call"]["arguments"]
                else:
                    # The generated text is inside the "content" key
                    generated_text = message["content"]

                # Extract the JSON object string from the generated text
                json_string = extract_json_string(generated_text)

                # Sanitize the JSON string and parse it as a JSON object
                data = await sanitize_json_string(json_string)

                if data is None:
                    # If JSON extraction or parsing fails, sanitize the raw HTML string
                    sanitized_html = clean_html(generated_text)
                    # Return the sanitized HTML as a dictionary so it can be passed to update_product_on_woocommerce
                    data = {"description": sanitized_html}

        except Exception as error:
            logger.error(f"Error during OpenAI API call: {error}")
            # If JSON extraction or parsing fails, sanitize the raw HTML string
            sanitized_html = clean_html(generated_text)
            # Return the sanitized HTML as a dictionary so it can be passed to update_product_on_woocommerce
            data = {"description": sanitized_html}

    return data


async def retry_failed_products(session):
    query = "SELECT id, product_name, product_url, short_description, description FROM failed_products"
    failed_products = await conn.fetch(query)
    print("Retrying failed products...")
    total_products = len(failed_products)
    elapsed_time = time.time() - start_time
    tasks = [
        update_product_descriptions_locally(
            session,  # pass session here
            wcapi,
            {
                "id": product[0],
                "name": product[1],
                "product_url": product[2],
                "short_description": product[3],
                "description": product[4],
            },
            index + 1,
            total_products,
            start_time,
            elapsed_time,
        )
        for index, product in enumerate(failed_products)
    ]
    await asyncio.gather(*tasks)


async def is_product_updated(product_id):
    query = "SELECT EXISTS(SELECT 1 FROM updated_products WHERE id = $1)"
    result = await conn.fetchval(query, product_id)
    return result


async def generate_seo_content(prompt, function_name, functions):
    assistant_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=assistant_messages,
        function_call={"name": function_name},
        functions=functions,  # This line is added to pass the functions to the API
    )
    print(f"Response: {response}")  # For debugging

    # Check that the response is not None and contains the expected data
    if response and response.get("choices") and len(response["choices"]) > 0:
        message = response["choices"][0]["message"]

        if "function_call" in message:
            # The generated text is inside the "arguments" key in the "function_call" object
            generated_text = message["function_call"]["arguments"]
        else:
            # The generated text is inside the "content" key
            generated_text = message["content"]

        # Extract the JSON object string from the generated text
        json_string = extract_json_string(generated_text)

        # Sanitize the JSON string and parse it as a JSON object
        data = await sanitize_json_string(json_string)

        if data is None:
            # If JSON extraction or parsing fails, sanitize the raw HTML string
            sanitized_html = clean_html(generated_text)
            # Return the sanitized HTML as a dictionary so it can be passed to update_product_on_woocommerce
            data = {"description": sanitized_html}

        return data

    return None  # Return None if the response is not as expected


async def update_product_descriptions_locally(
    session, wcapi, product, index, total, start_time, elapsed_time
):
    start_time = time.time()  # Capture the start time at the beginning of the function
    try:
        product_id = product["id"]
        product_title = product.get("name")
        short_description = product.get("short_description")
        if short_description is None:
            short_description = "No short description available."
        long_description = product.get("description")
        final_product_url = None

        # Logging the start of processing for this product
        logger.info(f"Starting update for product {product_id}: {product_title}")

        # Define the functions for GPT-4
        FUNCTIONS = [
            {
                "name": "generate_short_description",
                "description": "Generate a short description for a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "short_description": {"type": "string"},
                        "product_title": {"type": "string"},
                    },
                    "required": ["short_description", "product_title"],
                },
            },
            {
                "name": "generate_long_description",
                "description": "Generate a long description for a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "long_description": {"type": "string"},
                        "product_title": {"type": "string"},
                    },
                    "required": ["long_description", "product_title"],
                },
            },
            {
                "name": "generate_seo_content",
                "description": "Generate SEO content for a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_title": {"type": "string"},
                        "long_description": {"type": "string"},
                    },
                    "required": ["product_title", "long_description"],
                },
            },
        ]
        # Fetch the original product data
        original_product_data = await get_product_data(
            session, wcapi, product_id
        )  # pass session here

        # Generate prompts for short and long description
        short_desc_prompt = (
            f"As a product analyst, your task is to revamp our product listings for our new B2B marketplace. "
            f"Start with the base description: '{short_description}'. If the product has a chemical formula or CAS number, integrate "
            f"pertinent details from your data up to September 2021. For other products in the marketplace, use your general product knowledge "
            f"to enhance and expand the listing. Filter out irrelevant supplier data and unnecessary product specifics. "
            f"Craft a compelling 150-word short description in HTML for product '{product_title}'. Employ bullet points when presenting crucial details, "
            f"apply bold formatting to emphasize keywords in each bullet point, and omit any nonessential supplier specifics or irrelevant product data. "
            f"IMPORTANT: If specific data for an attribute like CAS number or chemical formula isn't known, SKIP that attribute entirely. Do not use placeholders like 'Not available' or 'still being determined'. "
            f'Please return ONLY a JSON object with the following key: "short_description". No additional text or formatting is needed.'
        ).format(short_description=short_description, product_title=product_title)

        # Call the generate_short_description function
        result_data_short = await call_gpt4_function(
            short_desc_prompt, "generate_short_description", FUNCTIONS
        )
        print(f"result_data_short: {result_data_short}")  # <-- Log the entire output
        if result_data_short is None:
            logger.error(
                f"No short description generated for product '{product_title}'. Skipping update."
            )
            return
        else:
            short_description = result_data_short["short_description"]
            print(f"Short Description: {short_description}")

        # Modify long description prompt
        long_desc_prompt = (
            f"As a product analyst, your task is to revamp our product listings for our new B2B marketplace. "
            f"For products that have CAS or a chemical formula, you can look for those attributes from the following list if available: "
            f"Compound Name, Synonyms, IUPAC Name, CAS Number, Molecular Formula, Molecular Weight, Canonical SMILES, InChI strings, "
            f"Boiling Point, Melting Point, Flash Point, Density, Solubility, Vapor Pressure, Refractive Index, pH, LogP, Polar Surface Area, "
            f"Rotatable Bond Count, Hazard and Precautionary Statements, GHS Classification, LD50, Routes of Exposure, Carcinogenicity, "
            f"Teratogenicity, Bioassay Results, Target Proteins, Mechanism of Action, Pharmacological Class, ADME data, NMR, MS, IR, UV-Vis, "
            f"Environmental Fate, Biodegradability, Ecotoxicity, Therapeutic Uses, Dosage, Contraindications, Side Effects, Drug Interactions.\n\n"
            f"For other products in the marketplace, use your general product knowledge data up to September 2021 to help enhance and expand the listing. "
            f"Start by utilizing the initial description provided: '{long_description}'. After gathering the details, craft a compelling 1000-word "
            f"detail product description in HTML highlighting the product's key features for '{product_title}'. Use bullet points for vital information."
            f"IMPORTANT: If specific data for an attribute isn't known, SKIP that attribute entirely. Do not use placeholders like 'Not available'.\n\n"
            f'Please return ONLY a JSON object with the following key: "long_description". No additional text or formatting is needed.'
        ).format(long_description=long_description, product_title=product_title)

        # Call the generate_long_description function
        result_data_long = await call_gpt4_function(
            long_desc_prompt, "generate_long_description", FUNCTIONS
        )
        print(f"result_data_long: {result_data_long}")  # <-- Log the entire output
        if result_data_long is None:
            logger.error(
                f"No long description generated for product '{product_title}'. Skipping update."
            )
            return
        else:
            long_description = result_data_long["long_description"]
            long_description = remove_control_characters_and_unwanted_backslashes(
                long_description
            )
            print(f"Long Description: {long_description}")

            # Sanitize the long description here
            long_description = remove_control_characters_and_unwanted_backslashes(
                long_description
            )

        # Prepare the required parameters for the prompt
        seo_content_prompt = {
            "product_title": product_title,  # Your actual product title
            "long_description": long_description,  # Your actual long description that will be used for the SEO content generation
        }

        # Construct the prompt message
        prompt_message = (
            f"Given the product title '{product_title}' and the long description:\n\n'{long_description}'\n\n"
            "Generate the following SEO content:\n"
            "- An SEO-friendly title (seo_title) for the product.\n"
            "- A meta description (meta_description) that summarizes the product in no more than 155 characters.\n"
            "- Two focus keywords (focus_keywords) that are central to the product's identity.\n"
            "- A few tags (tags) that describe key aspects of the product.\n"
            "- The broader categories (categories) that the product falls under.\n"
            "Format your response as a JSON object."
        )

        # Print the prompt message for debugging
        print(f"Prompt for SEO content generation: {prompt_message}")

        # Call the generate_seo_content function
        result_data_seo = await call_gpt4_function(
            prompt_message, "generate_seo_content", FUNCTIONS
        )
        print(f"result_data_seo: {result_data_seo}")  # <-- Log the entire output
        print(f"Response from generate_seo_content: {result_data_seo}")

        # Now unpack and print the result data if any
        if result_data_seo is None:
            logger.error(
                f"No SEO content generated for product '{product_title}'. Skipping update."
            )
            return
        elif "raw_response" in result_data_seo:
            # If JSON extraction failed, log the error
            logger.warning(
                f"Failed to extract SEO content for product '{product_title}'. Raw response: {result_data_seo['raw_response']}"
            )
        else:
            print(f"result_data_seo json: {result_data_seo}")
            seo_title = result_data_seo.get("seo_title")
            seo_meta_description = result_data_seo.get("meta_description")
            focus_keywords = result_data_seo.get("focus_keywords")
            tags = result_data_seo.get("tags")
            categories = result_data_seo.get("categories")

        # Fetch the category data
        all_categories = await get_categories(
            session, wcapi
        )  # Pass session and wcapi as arguments

        # Check if the generated categories are valid
        valid_categories = [cat for cat in categories if cat in all_categories]

        # Get the category ids based on the generated categories
        generated_categories = result_data_seo.get("categories", [])
        category_mapping = {cat["name"]: cat["id"] for cat in all_categories}
        new_category_ids = [
            {"id": category_mapping[cat]}
            for cat in generated_categories
            if cat in category_mapping
        ]

        # Check if there are less than 1 or more than 3 categories, and adjust if needed
        if not new_category_ids:
            # If no valid category is selected, manually select the primary category
            new_category_ids.append(
                {"id": all_categories[0]["id"]}
            )  # Assuming the primary category is the first one
        if len(new_category_ids) > 3:
            # If more than 3 categories are selected, trim it down to 3
            new_category_ids = new_category_ids[:3]

        # Update the product on WooCommerce
        response_data = await update_product_on_woocommerce(
            session,
            wcapi,
            product_id,
            short_description,
            long_description,
            new_category_ids,
            seo_title,
            seo_meta_description,
            tags,
            focus_keywords,
        )

        if response_data is None or not isinstance(response_data, dict):
            logger.error(
                f"Unexpected type for response_data: {type(response_data)}. Expected dict."
            )
            return
        final_product_url = response_data.get("permalink")

        # Calculate elapsed_time right after the update process is done
        elapsed_time = time.time() - start_time

        success = final_product_url is not None

        if success:
            print(
                f"Product Name: {product_title}\nTime Elapsed: {elapsed_time} seconds\nFinal Product URL: {final_product_url}\n"
            )
            logger.info(
                f"Product Name: {product_title}\nTime Elapsed: {elapsed_time} seconds\nFinal Product URL: {final_product_url}"
            )
            await save_updated_product(product_id, product_title, final_product_url)
        else:
            logger.error(
                f"Error when updating product: The product update operation was not successful."
            )
            await save_failed_product(
                product_id,
                product_title,
                final_product_url,
                short_description,
                long_description,
            )
    except Exception as e:
        logger.error(
            f"Error updating product {product['id']} - {product.get('name')}: {str(e)}"
        )
        await save_failed_product(
            product["id"],
            product.get("name"),
            product.get("permalink"),
            product.get("short_description"),
            product.get("description"),
        )

    # Return the elapsed time
    return elapsed_time


async def update_product_on_woocommerce(
    session,
    wcapi,
    product_id,
    new_short_description,
    new_long_description,
    categories,
    seo_title,
    seo_meta_description,
    tags,
    focus_keywords,
):
    # Fetch and print the original categories before the update
    original_product_data = await get_product_data(session, wcapi, product_id)
    original_categories = [
        category["name"] for category in original_product_data.get("categories", [])
    ]
    print(
        f"Original Categories for Product {product_id}: {', '.join(original_categories)}"
    )

    # Get the seo_title and sanitize it
    seo_title = original_product_data.get("seo_title")
    import re

    seo_title = re.sub(r"[\t\n\r]", "", seo_title)
    seo_title = seo_title.replace("\\", "\\\\")

    # Prepare the tags and categories in the format WooCommerce expects
    formatted_tags = [{"name": tag} for tag in tags]
    formatted_categories = [{"id": cat} for cat in categories]

    retries = 5  # Increased number of retries
    response_data = None
    backoff_factor = 0.3

    new_short_description = unescape(new_short_description)
    new_long_description = unescape(new_long_description)

    # Rate limiting: Introducing a delay between successive requests to avoid overwhelming the server
    await asyncio.sleep(2)

    for i in range(retries):
        try:
            # Update product data
            async with session.put(
                f"{wcapi.url}/wp-json/wc/v3/products/{product_id}",
                auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
                json={
                    "name": seo_title,
                    "short_description": new_short_description,
                    "description": new_long_description,
                    "categories": formatted_categories,
                    "tags": formatted_tags,
                    "meta_data": [
                        {"key": "_yoast_wpseo_metadesc", "value": seo_meta_description},
                        {"key": "_yoast_wpseo_title", "value": seo_title},
                        {"key": "_yoast_wpseo_focuskw", "value": focus_keywords},
                    ],
                },
                ssl=False,
            ) as resp:
                # Check if the response is HTML and handle it
                if "text/html" in resp.headers.get("Content-Type", ""):
                    logger.error(
                        f"Received an HTML response from WooCommerce API on attempt {i+1}: {await resp.text()}"
                    )
                    # If it's a gateway timeout error, we retry
                    if "Gateway time-out" in await resp.text():
                        raise Exception("Gateway timeout error.")
                    else:
                        return {}

                response_data = await resp.json()

                if resp.status == 200:
                    if response_data is None or not isinstance(response_data, dict):
                        logger.error(
                            f"Unexpected type for response_data: {type(response_data)}. Expected dict."
                        )
                        return {}
                    final_product_url = response_data.get("permalink")

                    # Print the updated categories after a successful update
                    updated_categories = [
                        category["name"]
                        for category in response_data.get("categories", [])
                    ]
                    print(
                        f"Updated Categories for Product {product_id}: {', '.join(updated_categories)}"
                    )

                    await remove_failed_product(product_id)
                    await save_updated_product(
                        product_id, response_data["name"], final_product_url
                    )
                    break  # If successful, break out of the retry loop
                else:
                    raise Exception(f"Error {resp.status}: {response_data}")

        except aiohttp.client_exceptions.ServerDisconnectedError:
            if i < retries - 1:
                await asyncio.sleep(backoff_factor * (2**i))  # Exponential backoff
                continue
            else:
                logger.error(f"Server disconnected after {i + 1} attempts.")
                response_data = {}  # Ensure response_data is a dictionary
                raise
        except Exception as e:
            logger.error(f"Error while updating the product on attempt {i+1}: {e}.")
            if i == retries - 1:
                await save_failed_product(
                    product_id,
                    "product_name",
                    "product_url",
                    new_short_description,
                    new_long_description,
                )
                return {}
            else:
                # Exponential backoff
                await asyncio.sleep(backoff_factor * (2**i))
                continue

    return response_data or {}  # Ensure response_data is a dictionary


async def get_product_data(session, wcapi, product_id):
    product_url = f"{wcapi.url}/wp-json/wc/v3/products/{product_id}"

    try:
        async with session.get(
            product_url,
            params={
                "consumer_key": wcapi.consumer_key,
                "consumer_secret": wcapi.consumer_secret,
            },
            timeout=10,  # Adjust the timeout value as needed
            ssl=False,
        ) as response:
            if response.status == 200:
                product_data = await response.json()
                # print(product_data)  # Commented out this line
                return product_data
            else:
                print(
                    f"Error while getting product data. Status code: {response.status}"
                )
    except asyncio.TimeoutError:
        print(f"Timeout while getting product data: {product_id}")
    except aiohttp.ClientError as e:
        print(f"Error while getting product data: {e}")

    return None


async def update_yoast_meta(session, wcapi, product_id, meta_key, meta_value):
    async with session.post(
        f"{wcapi.url}/wp-json/wp/v2/products/{product_id}/meta",
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        json={"meta_key": meta_key, "meta_value": meta_value},
        ssl=False,
    ) as seo_resp:
        if seo_resp.status != 200:
            logger.error(f"Error while updating {meta_key} for product {product_id}")
        else:
            print(f"Successfully updated {meta_key} for product {product_id}")


async def save_updated_product(product_id, product_name, product_url):
    print(
        f"Saving Product: ID={product_id}, Name={product_name}, URL={product_url}"
    )  # Print the product details before saving
    query = """
        INSERT INTO updated_products (id, product_name, product_url)
        VALUES ($1, $2, $3)
        ON CONFLICT (id) 
        DO UPDATE SET product_name = $2, product_url = $3
    """
    try:
        await conn.execute(query, product_id, product_name, product_url)
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error: {str(e)}")


async def view_recent_updates():
    query = "SELECT id, product_name, product_url, last_update FROM updated_products ORDER BY last_update DESC"
    products = await conn.fetch(query)
    print("Recently updated products:")
    for product in products:
        print(
            f"Product ID: {product['id']}, Product Name: {product['product_name']}, Product URL: {product['product_url']}, Last Update: {product['last_update']}"
        )


async def save_failed_product(
    product_id, product_name, product_url, short_description, description
):
    query = "INSERT INTO failed_products (id, product_name, product_url, short_description, description) VALUES ($1, $2, $3, $4, $5)"
    try:
        await conn.execute(
            query, product_id, product_name, product_url, short_description, description
        )
    except asyncpg.exceptions.UniqueViolationError:
        # Ignore the error if the record already exists
        pass
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error: {str(e)}")


async def remove_failed_product(product_id):
    query = "DELETE FROM failed_products WHERE id = $1"
    try:
        await conn.execute(query, product_id)
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error: {str(e)}")


async def get_products_in_category(session, wcapi, category_id, page, per_page=25):
    products = []

    async with semaphore, session.get(
        f"{wcapi.url}/wp-json/wc/v3/products",
        params={"category": category_id, "page": page, "per_page": per_page},
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        ssl=False,
    ) as resp:
        page_products = await resp.json()
        products.extend(page_products)

    return products


async def get_seconds_elapsed():
    query = "SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_update)) FROM updated_products"
    result = await conn.fetchval(query)
    return result


async def update_seconds_elapsed(product_id, product_url):
    query = "UPDATE updated_products SET last_update = CURRENT_TIMESTAMP, product_url = $1 WHERE id = $2"
    await conn.execute(query, product_url, product_id)


async def get_failed_products():
    query = "SELECT id, product_name, product_url FROM failed_products"
    return await conn.fetch(query)


async def update_one_product(session):
    product_id = input("Enter the ID of the product to update: ")

    # Start the timer here
    start_time = time.time()

    product_data = await get_product_data(
        session, wcapi, product_id
    )  # pass session here

    if not product_data:
        print("Product not found.")
        return

    product = {
        "id": product_id,
        "name": product_data.get("name"),
        "short_description": product_data.get("short_description"),
        "description": product_data.get("description"),
    }

    # Initialize elapsed_time
    elapsed_time = 0

    tasks = [
        update_product_descriptions_locally(
            session, wcapi, product, 1, 1, start_time, elapsed_time
        )  # Pass the start_time and elapsed_time to the function
    ]
    await asyncio.gather(*tasks)

    # Stop the timer and print the elapsed time here
    elapsed_time = time.time() - start_time
    print(f"Total time for updating product '{product_id}': {elapsed_time:.2f} seconds")


async def update_category_products(session):
    category_id = input("Enter the ID of the category to update: ")
    category_name, product_count = await get_category_data(session, wcapi, category_id)
    if not category_name:
        print("Category not found.")
        return

    print(f"Updating products in category '{category_name}'...")

    products = []
    page = 1
    per_page = 25

    while len(products) < product_count:
        page_products = await get_products_in_category(
            session, wcapi, category_id, page, per_page
        )
        products.extend(page_products)
        page += 1

    total_products = len(products)
    elapsed_time = time.time() - start_time

    async def handle_update(product, index):
        try:
            await update_product_descriptions_locally(
                session,
                wcapi,
                product,
                index + 1,
                total_products,
                start_time,
                elapsed_time,
            )
        except Exception as e:
            logger.error(
                f"Error updating product {product['id']} - {product.get('name')}: {str(e)}"
            )
            await save_failed_product(
                product["id"],
                product.get("name"),
                product.get("permalink"),
                product.get("short_description"),
                product.get("description"),
            )

    tasks = [
        handle_update(product, index)  # Use handle_update here
        for index, product in enumerate(products)
    ]
    await asyncio.gather(*tasks)


async def update_entire_catalog(session):
    print("Updating entire catalog...")
    products = []
    page = 1
    per_page = 100

    while True:
        async with session.get(
            f"{wcapi.url}/wp-json/wc/v3/products",
            params={"page": page, "per_page": per_page},
            auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
            ssl=False,
        ) as resp:
            page_products = await resp.json()
            products.extend(page_products)

            if len(page_products) < per_page:
                break

            page += 1

    total_products = len(products)
    elapsed_time = time.time() - start_time

    tasks = [
        update_product_descriptions_locally(
            session,  # pass session here
            wcapi,
            product,
            index + 1,
            total_products,
            start_time,
            elapsed_time,
        )
        for index, product in enumerate(products)
    ]
    await asyncio.gather(*tasks)


async def resume_previous_update(session):
    updated_products = await get_updated_product_ids()
    print("Resuming previous update...")
    print(f"Total products to update: {len(updated_products)}")

    for product_id in updated_products:
        product_data = await get_product_data(session, wcapi, product_id)
        if not product_data:
            print(f"Product with ID {product_id} not found.")
            continue

        product = {
            "id": product_id,
            "name": product_data.get("name"),
            "short_description": product_data.get("short_description"),
            "description": product_data.get("description"),
        }

        await update_product_descriptions_locally(
            session, wcapi, product, 1, 1, start_time, 0
        )


async def view_failed_products():
    failed_products = await get_failed_products()
    print("Failed products:")
    for product in failed_products:
        print(
            f"Product ID: {product['id']}, Product Name: {product['product_name']}, Product URL: {product['product_url']}"
        )


async def sanitize_and_verify_responses():
    query = "SELECT id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags FROM generated_responses"
    responses = await conn.fetch(query)

    for response in responses:
        id = response[0]
        short_description = clean_html(response[1])
        long_description = clean_html(response[2])
        seo_title = response[3]
        seo_meta_description = response[4]
        focus_keywords = response[5]
        tags = response[6]

        # Remove the outermost curly braces and sanitize the JSON
        focus_keywords = (
            ",".join(await sanitize_json_string(response[5].strip("{}")))
            if response[5]
            else None
        )
        tags = (
            ",".join(await sanitize_json_string(response[6].strip("{}")))
            if response[6]
            else None
        )

        query = """
            UPDATE generated_responses
            SET short_description = $1, long_description = $2, seo_title = $3, seo_meta_description = $4, focus_keywords = $5, tags = $6
            WHERE id = $7
        """
        try:
            await conn.execute(
                query,
                short_description,
                long_description,
                seo_title,
                seo_meta_description,
                focus_keywords,
                tags,
                id,
            )
            print(f"Sanitized and updated product id: {id}")
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error: {str(e)}")


async def update_verified_responses_on_woocommerce(session):  # add session parameter
    query = "SELECT id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags FROM generated_responses"
    responses = await conn.fetch(query)

    for response in responses:
        product_id = response["id"]
        short_description = response["short_description"]
        long_description = response["long_description"]
        seo_title = response["seo_title"]
        seo_meta_description = response["seo_meta_description"]
        focus_keywords = response["focus_keywords"]
        tags = response["tags"].split(
            ","
        )  # assuming tags are stored as a comma-separated string

        await update_product_on_woocommerce(
            session,
            wcapi,
            product_id,
            short_description,
            long_description,
            seo_title,
            seo_meta_description,
            tags,
            focus_keywords,
        )


async def view_responses_to_update():
    query = "SELECT id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags FROM generated_responses"
    responses = await conn.fetch(query)

    print("\nResponses to Update:")
    for response in responses:
        print(f"\nProduct ID: {response['id']}")
        print(f"Short Description: {response['short_description']}")
        print(f"Long Description: {response['long_description']}")
        print(f"SEO Title: {response['seo_title']}")
        print(f"SEO Meta Description: {response['seo_meta_description']}")
        print(f"Focus Keywords: {response['focus_keywords']}")
        print(f"Tags: {response['tags']}")


async def sanitize_and_update_responses():
    # This function calls your two existing functions
    await view_responses_to_update()
    await sanitize_and_verify_responses()
    await update_verified_responses_on_woocommerce()


async def get_updated_product_ids():
    query = "SELECT id FROM updated_products"
    result = await conn.fetch(query)
    return [row["id"] for row in result]


async def main():
    async with aiohttp.ClientSession() as session:
        print("\nConnected to WooCommerce API.")
        while True:
            print("Select an option:")
            print("1. Update one product")
            print("2. Update a whole category of products")
            print("3. View failed products")
            print("4. Retry failed products")
            print("5. View recent updates")
            print("6. Update entire catalog")
            print("7. Resume previous update")
            print("8. Manage responses to update")
            print("0. Exit")

            option = input("Enter an option (1/2/3/4/5/6/7/8/0): ")

            if option == "1":
                await update_one_product(session)
            elif option == "2":
                await update_category_products(session)
            elif option == "3":
                await view_failed_products()
            elif option == "4":
                await retry_failed_products(session)
            elif option == "5":
                await view_recent_updates()
            elif option == "6":
                await update_entire_catalog(session)
            elif option == "7":
                await resume_previous_update(session)
            elif option == "8":
                while True:
                    print("\nSelect an option:")
                    print("1. View responses to update")
                    print("2. Sanitize and update responses")
                    print("0. Back to main menu")

                    sub_option = input("Enter an option (1/2/0): ")

                    if sub_option == "1":
                        await view_responses_to_update()
                    elif sub_option == "2":
                        await sanitize_and_verify_responses()
                        await update_verified_responses_on_woocommerce(session)
                    elif sub_option == "0":
                        break
                    else:
                        print("Invalid option. Please try again.")
            elif option == "0":
                break
            else:
                print("Invalid option. Please try again.")


if __name__ == "__main__":
    start_time = time.time()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_db_connection())
    loop.run_until_complete(main())
    loop.run_until_complete(conn.close())
