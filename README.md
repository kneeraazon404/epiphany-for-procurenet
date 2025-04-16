# ePiphany-for-Procurenet  

## Overview  

**ePiphany-for-Procurenet** is a robust and efficient script designed to streamline product management in WooCommerce stores. Focused on improving SEO, handling errors gracefully, and optimizing performance through concurrency, this tool is essential for store administrators looking for automation and precision.  

---

## ✨ Features  

### **1. Database Management**  
- Connects seamlessly to a **PostgreSQL** database.  
- Automatically creates tables:  
  - **`updated_products`**: Tracks successfully updated products.  
  - **`failed_products`**: Logs products that encountered issues during updates.  

### **2. Interactive User Interface**  
- Offers a user-friendly and interactive console interface via the `main_menu()` function.  
- Simplifies navigation and operation for users.  

### **3. OpenAI Integration**  
- Enhances product descriptions using the **OpenAI API**.  
- The `generate_seo_content()` function provides enriched, SEO-optimized content to improve product visibility.  

### **4. WooCommerce API Integration**  
- Utilizes the **WooCommerce REST API** to:  
  - Retrieve product information.  
  - Update product details efficiently.  
- Built on asynchronous programming with `aiohttp` for smooth and fast API interactions.  

### **5. Asynchronous Concurrency with asyncio**  
- Built using Python's `asyncio` library for optimal performance.  
- Key functions include:  
  - `update_one_product()`: Updates a single product.  
  - `update_category_products()`: Updates products in a specific category.  
  - `update_entire_catalog()`: Updates the entire WooCommerce catalog concurrently.  

### **6. Error Handling and Retry Mechanism**  
- Comprehensive error-handling framework:  
  - Logs HTTP failures and saves them to the "failed_products" table.  
  - Allows for retrying failed updates at a later time.  

### **7. Logging**  
- Maintains detailed logs for:  
  - Successful product updates.  
  - Errors and failures.  
- Essential for debugging and progress tracking.  

### **8. Pagination Support**  
- Handles pagination efficiently when fetching product data from WooCommerce.  
- Ensures complete coverage of all products, even across multiple pages.  

---

## 🛠️ Technology Stack  

- **Programming Language**: Python  
- **Libraries**:  
  - `aiohttp`: For asynchronous HTTP requests.  
  - `asyncio`: For concurrency and performance optimization.  
  - OpenAI API: For generating SEO-optimized content.  
- **Databases**: PostgreSQL  
- **Platform**: WooCommerce  

---

## 🚀 Getting Started  

### Prerequisites  

Ensure you have the following installed:  
- Python 3.8 or above  
- PostgreSQL  
- WooCommerce store with API credentials  

### Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/kneeraazon404/ePiphany-for-Procurenet.git  
   cd ePiphany-for-Procurenet
   
2. Set up a virtual environment:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```  

3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

4. Configure environment variables for:  
   - PostgreSQL credentials  
   - WooCommerce API credentials  
   - OpenAI API key  

---

## ⚙️ Usage  

1. Run the script:  
   ```bash  
   python main.py  
   ```  

2. Follow the interactive menu to:  
   - Update a single product.  
   - Update products by category.  
   - Update the entire product catalog.  
   - Retry failed updates.  

---

## 📜 License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

---

## 🤝 Contributions  

Contributions are welcome! To contribute:  

1. Fork the repository.  
2. Create a feature branch:  
   ```bash  
   git checkout -b feature-name  
   ```  
3. Commit your changes:  
   ```bash  
   git commit -m "Add feature name"  
   ```  
4. Push to your fork:  
   ```bash  
   git push origin feature-name  
   ```  
5. Open a pull request detailing your changes.  

---

## 📧 Contact  

For support or inquiries:  
- **GitHub**: [kneeraazon404](https://github.com/kneeraazon404)  
- **Email**: kneeraazon@gmail.com  
```
