import asyncio

from aiohttp import ClientSession
from woocommerce import API

wcapi = API(
    url="https://procure-net.com",
    consumer_key="ck_302221e1ae92c8f9325f4d6de6b1da8aabf5353d",
    consumer_secret="cs_c49195e3967f050a758c6ddddd296b94648750cb",
    version="wc/v3",
)
semaphore = asyncio.Semaphore(25)
session = ClientSession()
