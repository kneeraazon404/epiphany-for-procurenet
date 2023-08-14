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
import openai
import psycopg2
import simplejson as json
from aiohttp import ClientSession, client_exceptions
from halo import Halo
from woocommerce import API

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


def delete_tables():
    c.execute("DELETE FROM updated_products")
    c.execute("DELETE FROM failed_products")
    conn.commit()


delete_tables()


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


def create_openai_responses_table():
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_responses (
            product_id INTEGER PRIMARY KEY,
            product_title TEXT,
            short_description TEXT,
            long_description TEXT,
            final_product_url TEXT
        )
        """
    )
    conn.commit()


# Call the function to ensure the table is created.
create_openai_responses_table()
logger.info("Table openai_responses created if not exist.")

create_tables()
logger.info(
    "PostgreSQL database connected and table updated_products created if not exist."
)


# Insert the generated product details into the openai_responses table
def insert_into_openai_responses(
    product_id, product_title, short_description, long_description, final_product_url
):
    try:
        c.execute(
            "INSERT INTO openai_responses (product_id, product_title, short_description, long_description, final_product_url) VALUES (%s, %s, %s, %s, %s)",
            (
                product_id,
                product_title,
                short_description,
                long_description,
                final_product_url,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting into openai_responses table: {e}")


# Call this function right after generating the product details


# 6. Implement the async functions and logic for product updates, WooCommerce API operations, and error handling
semaphore = asyncio.Semaphore(25)


def call_chat_completions(model, messages):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Handle potential rate limit errors



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


async def generate_description(prompt, product_title, description_type):
    prompt = remove_control_characters_and_unwanted_backslashes(prompt)
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
        except Exception as error:
            logger.error(f"Error during OpenAI API call: {error}")
            return None


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

        if await is_product_updated(product_id):
            response = input(
                f"Product '{product_id}' - {product_title} has already been updated. Do you want to reprocess it? (yes/no): "
            )
            if response.lower() != "yes":
                logger.info(f"Skipping product '{product_id}' - {product_title}.")
                return
            # Generate prompts for short and long description
        short_desc_prompt = (
            f"As a product analyst, your task is to update our product listings for a B2B marketplace that is tailored for bulk sales. "
            f"Start with the base description: '{short_description}'. If the product has a chemical formula or CAS number, integrate "
            f"pertinent details from your data up to September 2021. For other products in the marketplace, use your general product knowledge "
            f"to enhance and expand the listing. Filter out irrelevant supplier data and unnecessary product specifics. "
            f"Craft a compelling 150-word short description in HTML for product '{product_title}'. Employ bullet points when presenting crucial details, "
            f"apply bold formatting to emphasize keywords in each bullet point, and omit any nonessential supplier specifics or irrelevant product data. "
            f"IMPORTANT: If specific data for an attribute like CAS number or chemical formula isn't known, SKIP that attribute entirely. Do not use placeholders like 'Not available' or 'still being determined'. "
            f'Please return ONLY a JSON object with the following key: "short_description". No additional text or formatting is needed.'
        ).format(short_description=short_description, product_title=product_title)

        # Modify long description prompt
        long_desc_prompt = (
            f"As a product analyst, your task is to update our product listings for our B2B marketplace that is tailored for bulk sales. "
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
            f"Also, generate an SEO-friendly title for this product, a meta description that is under 155 characters, two focus keywords based on the meta description, and a list of relevant product tags.\n\n"
            f'Please return ONLY a JSON object with the following keys: "long_description", "seo_title", "meta_description", "focus_keywords", "tags". No additional text or formatting is needed.'
        ).format(long_description=long_description, product_title=product_title)
        result_data_short = await generate_description(
            short_desc_prompt, product_title, "short description"
        )

        if result_data_short is not None:
            short_description = result_data_short["short_description"]
        else:
            logger.error(
                f"No short description generated for product '{product_title}'. Skipping update."
            )
            return

        result_data_long = await generate_description(
            long_desc_prompt, product_title, "long description"
        )

        if result_data_long is not None:
            long_description = result_data_long["long_description"]
            seo_title_long = result_data_long["seo_title"]
            seo_meta_description_long = result_data_long["meta_description"]
            focus_keywords_long = result_data_long["focus_keywords"]
            tags_long = result_data_long["tags"]
        else:
            logger.error(
                f"No long description generated for product '{product_title}'. Skipping update."
            )
            return
        #  Insert the data into postgres db before updating the product on WooCommerce
        insert_into_openai_responses(
            product_id,
            product_title,
            short_description,
            long_description,
            final_product_url,
        )
        # Call the update_product_on_woocommerce with the appropriate arguments:

        response_data = await update_product_on_woocommerce(
            session,
            wcapi,
            product_id,
            short_description,
            long_description,
            seo_title_long,
            seo_meta_description_long,
            tags_long,
            focus_keywords_long,  # Pass the focus keywords
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


async def generate_seo_content(description):
    # Generate SEO title from the first sentence
    seo_title = description.split(".")[0]
    print(f"\nGenerated SEO Title:\n{seo_title}")

    # Generate SEO meta description from the first two sentences
    seo_meta_description = " ".join(description.split(".")[:2])
    print(f"\nGenerated SEO Meta Description:\n{seo_meta_description}")

    # Generate tags by splitting the description into words and picking the first few unique words
    tags = list(set(description.split()))[:10]
    print(f"\nGenerated Product Tags:\n{tags}")

    # Generate focus keywords by picking the first two unique words
    focus_keywords = " ".join(tags[:2])
    print(f"\nGenerated Focus Keywords:\n{focus_keywords}")

    return seo_title, seo_meta_description, tags, focus_keywords


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


async def update_product_on_woocommerce(
    session,
    wcapi,
    product_id,
    new_short_description,
    new_long_description,
    seo_title,
    seo_meta_description,
    tags,
    focus_keywords,
):
    # Prepare the tags in the format WooCommerce expects
    formatted_tags = [{"name": tag} for tag in tags]
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
                        return

                response_data = await resp.json()

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
                return None, None
            else:
                # Exponential backoff
                await asyncio.sleep(backoff_factor * (2**i))
                continue

    return response_data


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

    tasks = [handle_update(product, index) for index, product in enumerate(products)]
    await asyncio.gather(*tasks)


async def update_entire_catalog():
    print("Updating entire catalog...")
    products = []
    page = 1
    per_page = 25

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
            session, wcapi, product, index + 1, total_products, start_time, elapsed_time
        )
        for index, product in enumerate(products)
    ]
    await asyncio.gather(*tasks)


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
