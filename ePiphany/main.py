# 1. Import the necessary libraries
import asyncio
import time

import openai
from aiohttp import ClientSession
from halo import Halo

from database import c, conn
from utils import (
    resume_previous_update,
    retry_failed_products,
    update_category_products,
    update_entire_catalog,
    update_one_product,
    view_failed_products,
    view_recent_updates,
)

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
