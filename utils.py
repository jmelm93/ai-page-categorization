import os
import requests
import html2text
import pandas as pd
import asyncio
from typing import List
from loguru import logger
from requests.exceptions import RequestException, Timeout, HTTPError
from bs4 import BeautifulSoup

# Optional Proxy Env Vars
PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

def get_proxy_settings():
    """
    Returns a dictionary of proxy settings for requests, or None if no proxy is configured.
    """
    if PROXY_HOST and PROXY_USERNAME and PROXY_PASSWORD:
        return {
            "http": f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}",
            "https": f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}"
        }
    return None

def html_to_markdown(html_text: str) -> str:
    """
    Convert HTML to a simple markdown string using html2text.
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.bypass_tables = False
    return h.handle(html_text)

def extract_heading_blocks(html: str, heading_tags=('h1','h2','h3')) -> list[tuple[str, str]]:
    """
    Parse HTML and return a list of (heading_text, section_html).
    Each tuple covers the text *under* that heading until the next heading of equal or higher level.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find all heading elements of the specified types
    all_headings = soup.find_all(heading_tags)
    if not all_headings:
        return []

    heading_blocks = []
    
    for idx, heading in enumerate(all_headings):
        heading_text = heading.get_text(strip=True)

        # Determine which heading level (integer). e.g. <h2> => 2
        current_level = int(heading.name[-1])

        # We'll gather all HTML elements until the next heading of the same or higher level
        block_html_parts = []

        # Look at the elements following this heading, up to the next heading that is same/higher level
        for sibling in heading.next_siblings:
            # If it's another tag and is a heading, we check its level
            if sibling.name in heading_tags:
                next_level = int(sibling.name[-1])
                if next_level <= current_level:
                    # We reached a heading of the same or higher level; break
                    break

            # If it's not a heading (or a lower-level heading), include it
            if hasattr(sibling, "get_text"):
                block_html_parts.append(str(sibling))

        # Join the HTML parts into a single block
        block_html = "".join(block_html_parts).strip()
        if block_html:
            heading_blocks.append((heading_text, block_html))

    return heading_blocks


def convert_blocks_to_markdown(heading_blocks: list[tuple[str, str]]) -> list[str]:
    markdown_blocks = []
    for heading_text, block_html in heading_blocks:
        # Convert the heading itself (turn it into a Markdown heading)
        heading_md = f"# {heading_text}\n\n"  # or use "##" for H2, etc.
        
        # Convert the block HTML to Markdown
        block_md = html_to_markdown(block_html)
        
        # Combine them
        combined_md = heading_md + block_md
        markdown_blocks.append(combined_md)
    return markdown_blocks

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """
    Splits the text into multiple substrings if it exceeds `max_chars`.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

def chunk_markdown_blocks(markdown_blocks: list[str], max_chars: int = 2000) -> list[str]:
    """
    For each markdown block, create sub-chunks if needed.
    Returns a flat list of chunked strings.
    """
    final_chunks = []
    for md_block in markdown_blocks:
        # If the block is under the limit, just keep it
        if len(md_block) <= max_chars:
            final_chunks.append(md_block)
        else:
            # Otherwise, chunk it
            subchunks = chunk_text(md_block, max_chars=max_chars)
            final_chunks.extend(subchunks)
    return final_chunks

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
        # logger.info(f"Attempt direct request for {url}")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        markdown_text = html_to_markdown(resp.text)
        # logger.info(f"Direct request succeeded for {url}")
        return {"markdown": markdown_text, "html": resp.text} # Return raw HTML too
    except (RequestException, HTTPError, Timeout) as e:
        logger.warning(f"Direct request failed for {url}, checking proxy. {e}")

    # Attempt proxy fallback if set
    proxies = get_proxy_settings()
    if proxies:
        try:
            logger.info(f"Attempting proxy for {url}")
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=20)
            resp.raise_for_status()
            markdown_text = html_to_markdown(resp.text)
            # logger.info(f"Proxy request succeeded for {url}")
            return {"markdown": markdown_text, "html": resp.text} # Return raw HTML
        except (RequestException, HTTPError, Timeout) as e2:
            logger.error(f"Proxy request failed for {url}: {e2}")
            return {"markdown": "", "html": ""}
    else:
        logger.error("No proxy credentials found. Failing request.")
        return {"markdown": "", "html": ""}


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