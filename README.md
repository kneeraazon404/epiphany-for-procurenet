# ePiphany-for-Procurenet

## Overview
ePiphany-for-Procurenet is a powerful script designed to manage product updates in a WooCommerce store. It emphasizes SEO improvements, robust error handling, and efficient processing through concurrent operations to streamline the product updating process.

## Features

### **Database Connection & Tables Creation:**
- Connects to a **PostgreSQL** database.
- Automatically creates tables ("updated_products", "failed_products") if they don't exist.
- Tracks products that have been updated successfully or have faced issues during the update process.

### **User-friendly Menu Interface:**
- Offers an interactive console interface through the `main_menu()` function.
- Central to the script's operation and intuitive for users.

### **OpenAI API Integration:**
- Utilizes the **OpenAI API** to enhance product descriptions.
- The `generate_seo_content()` function is pivotal for generating enriched SEO content. (Note: The exact API call to OpenAI isn't detailed in the provided code).

### **WooCommerce API Integration:**
- Leverages the **WooCommerce REST API** for both retrieving and updating product details.
- Uses `aiohttp` for making asynchronous GET and PUT requests, ensuring smooth integration with the WooCommerce store.

### **Concurrency with Asyncio:**
- Built on Python's `asyncio` library for optimal performance.
- Functions like `update_one_product()`, `update_category_products()`, and `update_entire_catalog()` employ asynchronous programming for concurrent product processing.

### **Error Handling and Retry Mechanism:**
- Efficiently handles HTTP request failures using `aiohttp` and `asyncio`.
- In case of an update failure, the script logs the error, saves the product to the "failed_products" table, and provides an option to retry updating these products at a later time.

### **Logging:**
- Logs every significant event from updates to failures.
- Essential for debugging, tracking progress, and maintaining transparency.

### **Pagination Handling:**
- Manages pagination effectively when fetching product data from WooCommerce.
- Guarantees all products are fetched, even if distributed over multiple pages in WooCommerce's API.

## Conclusion

**ePiphany-for-Procurenet** is a comprehensive solution tailored for WooCommerce store managers. It integrates user-centric design with robust backend functions, making it an invaluable tool for seamless product management and SEO optimization.
