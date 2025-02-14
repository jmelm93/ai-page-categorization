import os
import asyncio
import requests
import html2text
import pandas as pd
import datetime
from typing import List
from loguru import logger
from requests.exceptions import RequestException, Timeout, HTTPError


# Optional Proxy Env Vars
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")


def html_to_markdown(html_text: str) -> str:
    """
    Convert HTML to a simple markdown string using html2text.
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.bypass_tables = False
    return h.handle(html_text)


def fetch_page_content(url: str) -> dict:
    """
    1) Attempt direct GET with Chrome user-agent.
    2) If fails, attempt the BrightData proxy if env vars are present.
    3) Convert HTML to Markdown on success, or return blank.
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    " AppleWebKit/537.36 (KHTML, like Gecko)"
                    " Chrome/103.0.0.0 Safari/537.36")
    }

    # Direct request
    try:
        logger.info(f"Attempt direct request for {url}")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        markdown_text = html_to_markdown(resp.text)
        logger.info(f"Direct request succeeded for {url}")
        return {"markdown": markdown_text}
    except (RequestException, HTTPError, Timeout) as e:
        logger.warning(f"Direct request failed for {url}, checking proxy. {e}")

    # Attempt proxy fallback if set
    if PROXY_HOST and PROXY_USERNAME and PROXY_PASSWORD:
        proxies = {
            "http": f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}",
            "https": f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}"
        }
        try:
            logger.info(f"Attempting proxy for {url}")
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=20)
            resp.raise_for_status()
            markdown_text = html_to_markdown(resp.text)
            logger.info(f"Proxy request succeeded for {url}")
            return {"markdown": markdown_text}
        except (RequestException, HTTPError, Timeout) as e2:
            logger.error(f"Proxy request failed for {url}: {e2}")
            return {"markdown": ""}
    else:
        logger.error("No proxy credentials found. Failing request.")
        return {"markdown": ""}


async def async_scrape_url(url: str) -> dict:
    """
    Non-blocking call to fetch_page_content by running in a thread.
    """
    return await asyncio.to_thread(fetch_page_content, url)


def load_urls_from_csv(filepath: str) -> List[str]:
    df = pd.read_csv(filepath, dtype=str)
    if "url" not in df.columns:
        logger.error("No 'url' column found in input CSV.")
        return []
    return df["url"].dropna().apply(str.strip).tolist()


# def write_csv(state) -> dict:
#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)
#     now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"page_groups_{now_str}.csv"
#     path = os.path.join(output_dir, filename)

#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(state["final_csv"])
#         logger.info(f"CSV written to {path}")
#     except Exception as e:
#         logger.error(f"Error writing CSV: {e}")
#     return {}