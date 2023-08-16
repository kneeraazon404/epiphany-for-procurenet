# 1. Import the necessary libraries
import asyncio
import json
import logging
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html import unescape
import aiohttp
import openai
import psycopg2
import simplejson as json
from aiohttp import ClientSession, client_exceptions
from halo import Halo
from woocommerce import API
import re

# Spinner setup
success_message = "Loading success"
failed_message = "Loading failed"
unicorn_message = "Loading unicorn"

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

# 2. Use the spinner for the loading effect

# API keys and other constants
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

wcapi = API(
    url="https://procure-net.com",
    consumer_key="ck_302221e1ae92c8f9325f4d6de6b1da8aabf5353d",
    consumer_secret="cs_c49195e3967f050a758c6ddddd296b94648750cb",
    version="wc/v3",
)

# 3. Set up API keys and other constants

# Logging setup
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

# 4. Set up the logging configuration

# Database connection and table creation

conn = psycopg2.connect(
    database="apppgsqldb",
    user="apppgsqluser",
    password="p0$tGrr3$qladM!n18072023",
    host="43.207.8.208",
    port="5432",
)
c = conn.cursor()


# Global variables for tracking progress
total_products = 0
processed_products = 0
failed_products = 0
start_time = 0


# This function runs as a separate asyncio task, updating the footer every second
async def update_footer():
    global total_products, processed_products, failed_products, start_time
    while True:
        elapsed_time = time.time() - start_time
        print("\033[K", end="\r")  # Clear the current line
        print(
            f"Time elapsed: {elapsed_time:.2f} seconds, Total products: {total_products}, Processed: {processed_products}, Failed: {failed_products}",
            end="\r",
        )
        await asyncio.sleep(1)


#  def delete_tables():
#      c.execute("DELETE FROM updated_products")
#      c.execute("DELETE FROM failed_products")
#      c.execute("DELETE FROM generated_responses")
#      conn.commit()

#  delete_tables()


def create_tables():
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS updated_products (
            id INTEGER PRIMARY KEY,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            product_name TEXT,
            product_url TEXT,
            retry_failed BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE IF NOT EXISTS failed_products (
            id INTEGER PRIMARY KEY,
            product_name TEXT,
            product_url TEXT
        )
    """
    )
    conn.commit()


create_tables()
logger.info(
    "PostgreSQL database connected and table updated_products created if not exist."
)

# 5. Connect to the PostgreSQL database and create necessary tables

# 6. Implement the async functions and logic for product updates, WooCommerce API operations, and error handling
semaphore = asyncio.Semaphore(25)


def call_chat_completions(model, messages):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Handle potential rate limit errors
    if not isinstance(response, dict):
        logger.error(
            "Unexpected response from OpenAI API. Waiting for 5 seconds before retrying..."
        )
        time.sleep(5)  # Wait for 5 seconds
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )

        time.sleep(5)  # Wait for 5 seconds before the next request
        return None

    # Check for rate limit errors in the API response
    if "usage" in response and "total_tokens" in response["usage"]:
        total_tokens = response["usage"]["total_tokens"]
        if total_tokens > 9000:  # You may need to adjust this threshold
            logger.warning("Approaching rate limit. Pausing for a moment.")
            time.sleep(10)  # Wait for 10 seconds

    print(f"Raw API Response: {response}")

    return response


def extract_json_string(generated_text):
    # Ensure there's at least one pair of '{' and '}'

    # Find all occurrences of '{' and '}'
    start_indices = [i for i, char in enumerate(generated_text) if char == "{"]
    end_indices = [i for i, char in enumerate(generated_text) if char == "}"]
    json_match = re.search(r"(\{.*?\})", generated_text)

    if json_match:
        # Extract the matched JSON string
        json_string = json_match.group(1)

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
    unwanted_chars = ["\a", "\f", "\v", "\n", "\\\\", "\\"]
    for char in unwanted_chars:
        s = s.replace(char, "")

    # Handling double backslashes and other escape sequences.
    sanitized_string = ""
    i = 0
    while i < len(s):
        if s[i] == "\\":
            # If double backslashes are found, replace them with a single backslash
            if i + 1 < len(s) and s[i + 1] == "\\":
                sanitized_string += "\\\\"
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


async def request_with_retry(url, retries=3, backoff_factor=0.3):
    for retry in range(retries):
        async with semaphore:  # Acquire the semaphore
            try:
                async with ClientSession() as session:
                    # Your request logic here
                    # For example:
                    response = await session.get(url)
                    response_data = await response.text()
                    return response_data
                # If successful, break out of loop
                break
            except client_exceptions.ServerDisconnectedError:
                if retry < retries - 1:  # if not the last retry
                    await asyncio.sleep(backoff_factor * (2**retry))
                else:
                    raise


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


async def retry_failed_products():
    query = "SELECT id, product_name, product_url, short_description, description FROM failed_products"
    c.execute(query)
    failed_products = c.fetchall()
    print("Retrying failed products...")
    total_products = len(failed_products)
    elapsed_time = time.time() - start_time
    tasks = [
        update_product_descriptions_locally(
            session,
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
    query = "SELECT EXISTS(SELECT 1 FROM updated_products WHERE id = %s)"
    c.execute(query, (product_id,))
    return c.fetchone()[0]


async def generate_description(prompt, product_title, description_type):
    prompt = remove_control_characters_and_unwanted_backslashes(prompt)
    assistant_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
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

            if response is None or not isinstance(response, dict):
                logger.error("Received None response from OpenAI API. Retrying...")
                return None

            if response.get("choices") and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]
                # print(f"Generated text: {generated_text}")  # Debug print

                # Extract the JSON object string from the generated text
                json_string = extract_json_string(generated_text)

                # Sanitize the JSON string and parse it as a JSON object
                data = sanitize_json_string(json_string)
                # print(f"Data: {data}")  # Debug print

                return data
        except Exception as error:
            logger.error(f"Error during OpenAI API call: {error}")
            return None


async def get_all_categories(session, wcapi):
    categories = []
    async with semaphore, session.get(
        f"{wcapi.url}/wp-json/wc/v3/products/categories",
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        ssl=False,
    ) as resp:
        categories = await resp.json()  # This should be a list of categories

    return categories  # Each category should be a dictionary with at least 'id' and 'name' fields


async def get_category_ids(session, wcapi, category_names):
    all_categories = await get_all_categories(session, wcapi)
    name_to_id = {
        cat["name"]: cat["id"] for cat in all_categories
    }  # Create a mapping of category names to IDs
    # print(f"name_to_id: {name_to_id}")  # Debug print
    # print(f"category_names: {category_names}")  # Debug print

    category_ids = []
    for name in category_names:
        id = name_to_id.get(name)  # Get the ID corresponding to the category name
        if id is not None:  # If the ID exists, append it to the list
            category_ids.append(id)

    # print(f"Category IDs: {category_ids}")  # Debug print

    return category_ids  # This should be a list of integers


async def generate_short_description(
    product_title, short_description, categories_string
):
    short_desc_prompt = (
        f"As a product analyst, your task is to update and create a compelling product listings for our B2B marketplace. "
        f"Start with the base description: '{short_description}'. If the product has a chemical formula or CAS number, integrate "
        f"pertinent details from your data up to September 2021. For other products in the marketplace, use your general product knowledge "
        f"to enhance and expand the listing. Filter out irrelevant supplier data and unnecessary product specifics. "
        f"Craft a compelling 150-word short description in HTML for product '{product_title}'. Employ bullet points when presenting crucial details, "
        f"apply bold formatting to emphasize keywords in each bullet point, and omit any nonessential supplier specifics or irrelevant product data. "
        f"IMPORTANT: If specific data for an attribute like CAS number or chemical formula isn't known, SKIP that attribute entirely. Do not use placeholders like 'Not available' or 'still being determined'. "
        f"Also, select up to three categories from the following list that the product could belong to: {categories_string}..."
        f'Please return ONLY a JSON object with the following keys: "short_description", "categories".'
        f'The "categories" should be a list of category IDs. No additional text or formatting is needed.'
    )
    return json.dumps({"short_description": short_desc_prompt})


async def generate_long_description(product_title, long_description):
    long_desc_prompt = (
        f"As a product analyst, your task is to update and create a compelling product listings for our B2B marketplace. "
        f"For products that have CAS or a chemical formula, you can look for those attributes from the following list if available: "
        f"Compound Name, Synonyms, IUPAC Name, CAS Number, Molecular Formula, Molecular Weight, Canonical SMILES, InChI strings, "
        f"Boiling Point, Melting Point, Flash Point, Density, Solubility, Vapor Pressure, Refractive Index, pH, LogP, Polar Surface Area, "
        f"Rotatable Bond Count, Hazard and Precautionary Statements, GHS Classification, LD50, Routes of Exposure, Carcinogenicity, "
        f"Teratogenicity, Bioassay Results, Target Proteins, Mechanism of Action, Pharmacological Class, ADME data, NMR, MS, IR, UV-Vis, "
        f"Environmental Fate, Biodegradability, Ecotoxicity, Therapeutic Uses, Dosage, Contraindications, Side Effects, Drug Interactions.\n\n"
        f"For other products in the marketplace, use your general product knowledge data up to September 2021 to help enhance and expand the listing. "
        f"Start by utilizing the initial description provided: '{long_description}'. After gathering the details, craft a compelling 1000-word "
        f"detail product description in HTML highlighting the product's key features for '{product_title}'. Use bullet points for vital information "
        f"and exclude any unrelated supplier specifics or non-pertinent product data.\n\n"
        f"IMPORTANT: If specific data for an attribute isn't known, SKIP that attribute entirely. Do not use placeholders like 'Not available'.\n\n"
        f"Also, generate an SEO-friendly title for this product, a meta description that is under 155 characters, a focus keyword based on the meta description, and a list of relevant product tags.\n\n"
        f'Please return ONLY a JSON object with the following keys: "long_description", "seo_title", "meta_description", "focus_keyword", "tags". No additional text or formatting is needed.'
    )
    return json.dumps({"long_description": long_desc_prompt})


async def update_product_descriptions_locally(
    session, wcapi, product, index, total, start_time, elapsed_time
):
    try:
        # Capture the start time right after fetching the product details from WooCommerce
        start_time = time.time()

        product_id = product["id"]
        product_title = product.get("name")
        short_description = product.get("short_description")
        if short_description is None:
            short_description = "No short description available."
        long_description = product.get("description")
        final_product_url = None
        response_data = {}

        if await is_product_updated(product_id):
#            response = input(
#                f"Product '{product_id}' - {product_title} has already been updated. Do you want to reprocess it? (yes/no): "
#            )
#            if response.lower() != "yes":
                logger.info(f"Skipping product '{product_id}' - {product_title}.")
                return

        # Fetch all categories
        all_categories = await get_all_categories(session, wcapi)
        categories_string = ", ".join(
            f"{cat['id']}: {cat['name']}" for cat in all_categories
        )

        # Use a ThreadPoolExecutor to run generate_short_description and generate_long_description concurrently
        with ThreadPoolExecutor() as executor:
            short_desc_prompt_future = executor.submit(
                generate_short_description,
                product_title,
                short_description,
                categories_string,
            )
            long_desc_prompt_future = executor.submit(
                generate_long_description, product_title, long_description
            )

        # Call result() on each future to ensure the function has completed and to get the result
        short_desc_prompt = short_desc_prompt_future.result()
        long_desc_prompt = long_desc_prompt_future.result()

        # For short description
        result_data_short = await generate_description(
            await short_desc_prompt, product_title, "short description"
        )
        category_ids = []  # Define category_ids as an empty list
        if result_data_short is not None:
            short_description = result_data_short["short_description"]
            categories = result_data_short.get(
                "categories", []
            )  # Get categories directly from the result_data_short
            if isinstance(categories, list):  # Check if categories is a list
                category_ids = categories
            elif isinstance(categories, int):  # Check if categories is a single integer
                category_ids = [categories]  # Convert the single integer into a list
            else:
                logger.error(
                    f"Categories for product '{product_title}' is neither a list nor a single integer."
                )
            if None in category_ids:
                category_ids.remove(None)  # Remove None values from the list

        # For long description
        result_data_long = await generate_description(
            await long_desc_prompt, product_title, "long description"
        )
        if result_data_long:
            long_description = result_data_long.get("long_description")
            seo_title_long = result_data_long.get("seo_title")
            seo_meta_description_long = result_data_long.get("meta_description")
            focus_keyword_long = result_data_long.get("focus_keyword")
            tags_long = result_data_long.get("tags")

        final_product_url = response_data.get("permalink")

        if result_data_long is not None:
            long_description = result_data_long["long_description"]
            seo_title_long = (
                result_data_long.get("seo_title") or ""
            )  # If seo_title_long is None, set it to an empty string
            if isinstance(seo_title_long, list):
                seo_title_long = " ".join(seo_title_long)
            seo_meta_description_long = result_data_long["meta_description"]
            focus_keyword_long = result_data_long["focus_keyword"]
            tags_long = result_data_long["tags"]

        response_data = await update_product_on_woocommerce(
            session,
            wcapi,
            product_id,
            short_description,
            long_description,
            seo_title_long,
            seo_meta_description_long,
            tags_long,
            category_ids,  # Pass the category IDs to the update function
            focus_keyword_long,
        )

        if response_data is None or not isinstance(response_data, dict):
            logger.error(
                f"Unexpected type for response_data: {type(response_data)}. Expected dict."
            )
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

    # return the necessary variables at the end of the function
    return product_id, short_description, long_description, seo_title_long, seo_meta_description_long, focus_keyword_long, tags_long, category_ids



async def update_product_on_woocommerce(
    session,
    wcapi,
    product_id,
    new_short_description,
    new_long_description,
    seo_title,
    seo_meta_description,
    tags,
    categories,
    focus_keyword,
):
    # Prepare the tags in the format WooCommerce expects
    formatted_tags = [{"name": tag} for tag in tags]
    retries = 5  # Increased number of retries
    response_data = None
    backoff_factor = 0.3

    new_short_description = unescape(new_short_description)
    new_long_description = unescape(new_long_description)

    # Rate limiting: Introducing a delay between successive requests
    await asyncio.sleep(2)

    timeout = aiohttp.ClientTimeout(total=60)  # Setting a 60 seconds timeout

    # Fetch all categories
    all_categories = await get_all_categories(session, wcapi)
    id_to_name = {
        cat["id"]: cat["name"] for cat in all_categories
    }  # Create a mapping of category IDs to names

    # Convert categories to a list of integers
    category_ids = [
        int(category_id) for category_id in categories if category_id is not None
    ]

    # Create formatted_categories using the category_ids
    formatted_categories = [{"id": category_id} for category_id in category_ids]

    # Add the print statements here
    print(f"seo_title: {seo_title}, type: {type(seo_title)}")
    print(
        f"formatted_categories: {formatted_categories}, type: {type(formatted_categories)}"
    )
    print(f"categories: {categories}, type: {type(categories)}")
    print(
        f"category_names: {[id_to_name.get(id) for id in category_ids]}"
    )  # Print the category names

    for i in range(retries):
        try:
            # Construct the payload
            payload = {
                "name": seo_title,
                "short_description": new_short_description,
                "categories": formatted_categories,
                "description": new_long_description,
                "tags": formatted_tags,
                "meta_data": [
                    {"key": "_yoast_wpseo_metadesc", "value": seo_meta_description},
                    {"key": "_yoast_wpseo_title", "value": seo_title},
                    {"key": "_yoast_wpseo_focuskw", "value": focus_keyword},
                ],
            }

            # Now, make the API call using this payload
            async with session.put(
                f"{wcapi.url}/wp-json/wc/v3/products/{product_id}",
                auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
                json=payload,
                ssl=False,
                timeout=timeout,
            ) as resp:
                response_data = await resp.json()

                # Check if response_data is a dictionary
                if not isinstance(response_data, dict):
                    logger.error(
                        f"Unexpected data type: {type(response_data)}. Expected dict."
                    )
                    return

                if resp.status == 200:
                    if response_data is None or not isinstance(response_data, dict):
                        logger.error(
                            f"Unexpected type for response_data: {type(response_data)}. Expected dict."
                        )
                        return
                    final_product_url = response_data.get("permalink")
                    await remove_failed_product(product_id)
                    await save_updated_product(
                        product_id, response_data["name"], final_product_url
                    )
                    break  # If successful, break out of the retry loop
                else:
                    raise Exception(f"Error {resp.status}: {response_data}")

        except asyncio.TimeoutError:
            logger.error(
                f"Request to update product {product_id} timed out on attempt {i+1}."
            )
            if i == retries - 1:
                await save_failed_product(
                    product_id,
                    "product_name",
                    "product_url",
                    new_short_description,
                    new_long_description,
                )
                return None, None
            else:
                # Exponential backoff
                await asyncio.sleep(backoff_factor * (2**i))
                continue
        except aiohttp.client_exceptions.ServerDisconnectedError:
            if i < retries - 1:
                await asyncio.sleep(backoff_factor * (2**i))  # Exponential backoff
                continue
            else:
                logger.error(f"Server disconnected after {i + 1} attempts.")
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
                await save_generated_response(
                    product_id,
                    new_short_description,
                    new_long_description,
                    seo_title,
                    seo_meta_description,
                    focus_keyword,
                    tags,
                    categories,
                )
                return None, None
            else:
                # Exponential backoff
                await asyncio.sleep(backoff_factor * (2**i))
                continue

    return response_data

async def save_generated_response(product_id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags, categories):
    query = """
        INSERT INTO generated_responses (id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags, categories)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:  # Create a new cursor for this query
            cur.execute(
                query,
                (product_id, short_description, long_description, seo_title, seo_meta_description, focus_keywords, tags, categories),
            )
            conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()

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
        VALUES (%s, %s, %s)
        ON CONFLICT (id) 
        DO UPDATE SET product_name = %s, product_url = %s
    """
    try:
        with conn.cursor() as cur:  # Create a new cursor for this query
            cur.execute(
                query,
                (product_id, product_name, product_url, product_name, product_url),
            )
            conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()


async def view_recent_updates():
    query = "SELECT id, product_name, product_url, last_update FROM updated_products ORDER BY last_update DESC"
    c.execute(query)
    products = c.fetchall()
    print("Recently updated products:")
    for product in products:
        print(
            f"Product ID: {product[0]}, Product Name: {product[1]}, Product URL: {product[2]}, Last Update: {product[3]}"
        )


async def save_failed_product(
    product_id, product_name, product_url, short_description, description
):
    query = "INSERT INTO failed_products (id, product_name, product_url, short_description, description) VALUES (%s, %s, %s, %s, %s)"
    try:
        c.execute(
            query,
            (product_id, product_name, product_url, short_description, description),
        )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        # Ignore the error if the record already exists
        conn.rollback()
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()  # Roll back the transaction if an error occurs


async def remove_failed_product(product_id):
    query = "DELETE FROM failed_products WHERE id = %s"
    try:
        with conn.cursor() as cur:  # Create a new cursor for this query
            cur.execute(query, (product_id,))
            conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()


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
    c.execute(query)
    return c.fetchone()[0]


async def update_seconds_elapsed(product_id, product_url):
    query = "UPDATE updated_products SET last_update = CURRENT_TIMESTAMP, product_url = %s WHERE id = %s"
    c.execute(query, (product_url, product_id))
    conn.commit()


async def get_failed_products():
    query = "SELECT id, product_name, product_url FROM failed_products"
    c.execute(query)
    return c.fetchall()


async def update_one_product():
    product_id = input("Enter the ID of the product to update: ")

    # Start the timer here
    start_time = time.time()

    product_data = await get_product_data(session, wcapi, product_id)
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


async def update_category_products():
    category_id = input("Enter the ID of the category to update: ")
    category_name, product_count = await get_category_data(session, wcapi, category_id)
    if not category_name:
        print("Category not found.")
        return

    print(f"Updating products in category '{category_name}'...")

    products = []
    page = 1
    per_page = 100

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

    tasks = [handle_update(product, index) for index, product in enumerate(products)]
    await asyncio.gather(*tasks)


async def get_all_products(session, wcapi, page, per_page=100):
    products = []
    async with session.get(
        f"{wcapi.url}/wp-json/wc/v3/products",
        params={"page": page, "per_page": per_page},
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        ssl=False,
    ) as resp:
        products_data = await resp.json()
        products.extend(products_data)
    return products

async def get_total_products(session, wcapi):
    global total_products  # declare total_products as global

    async with session.get(
        f"{wcapi.url}/wp-json/wc/v3/products",
        params={"page": 1, "per_page": 1},
        auth=aiohttp.BasicAuth(wcapi.consumer_key, wcapi.consumer_secret),
        ssl=False,
    ) as resp:
        total_products = int(resp.headers.get("X-WP-Total", 0))  # set the global total_products variable

    return total_products

async def update_entire_catalog():
    print("Updating entire catalog...")
    page = 1
    per_page = 100

    # Ask user which set of product IDs to update: odd or even
    id_set = input("Enter which set of product IDs to update (odd/even): ")
    
    if id_set.lower() not in ("odd", "even"):
        print("Invalid input. Defaulting to odd set.")
        id_set = "odd"

    # Fetch all products
    all_products = []
    while True:
        products = await get_all_products(session, wcapi, page, per_page)
        if not products:  # If no more products, break the loop
            break
        all_products.extend(products)
        page += 1

    # Filter products based on the user-provided set (odd or even IDs)
    if id_set == "odd":
        all_products = [prod for prod in all_products if prod['id'] % 2 != 0]
    else:
        all_products = [prod for prod in all_products if prod['id'] % 2 == 0]

    total_products = len(all_products)

    tasks = []
    for index, product in enumerate(all_products):
        elapsed_time = time.time() - start_time  # Calculate elapsed_time here
        tasks.append(handle_update(product, index, elapsed_time))
    await asyncio.gather(*tasks)

async def handle_update(product, index, elapsed_time):
    global processed_products, failed_products  # Declare these as global inside the function
    try:
        # capture the returned variables from update_product_descriptions_locally()
        product_id, short_description, long_description, seo_title_long, seo_meta_description_long, focus_keyword_long, tags_long, category_ids = await update_product_descriptions_locally(
            session,
            wcapi,
            product,
            index + 1,
            total_products,
            start_time,
            elapsed_time,
        )
        processed_products += 1  # Increment processed_products after a successful update
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
        failed_products += 1  # Increment failed_products after a failed update
        # Save the generated response when the product update fails
        await save_generated_response(
            product_id,
            short_description,
            long_description,
            seo_title_long,
            seo_meta_description_long,
            focus_keyword_long,
            tags_long,
            category_ids,
        )

async def resume_previous_update():
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
    query = "SELECT id, product_name, product_url FROM failed_products"
    c.execute(query)
    failed_products = c.fetchall()
    print("Failed products:")
    for product in failed_products:
        print(
            f"Product ID: {product[0]}, Product Name: {product[1]}, Product URL: {product[2]}"
        )


async def get_updated_product_ids():
    query = "SELECT id FROM updated_products"
    c.execute(query)
    return [row[0] for row in c.fetchall()]


async def main_menu():
    # Start the footer update task
    asyncio.create_task(update_footer())
    while True:
        print("\nConnected to WooCommerce API.")
        print("Select an option:")
        print("1. Update one product")
        print("2. Update a whole category of products")
        print("3. View failed products")
        print("4. Retry failed products")
        print("5. View recent updates")
        print("6. Update entire catalog")
        print("7. Resume previous update")
        print("0. Exit")

        option = input("Enter an option (1/2/3/4/5/6/7/0): ")

        if option == "1":
            await update_one_product()
        elif option == "2":
            await update_category_products()
        elif option == "3":
            await view_failed_products()
        elif option == "4":
            await retry_failed_products()
        elif option == "5":
            await view_recent_updates()
        elif option == "6":
            await update_entire_catalog()
        elif option == "7":
            await resume_previous_update()
        elif option == "0":
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    start_time = time.time()

    session = ClientSession()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_menu())

    session.close()
    c.close()
    conn.close()
