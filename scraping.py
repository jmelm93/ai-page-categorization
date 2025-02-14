import re
import asyncio
from typing import List
from loguru import logger
from pydantic import ValidationError
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from models import PageSegmentation, FlowState
from utils import (
    async_scrape_url,
    extract_heading_blocks,      
    convert_blocks_to_markdown, 
    chunk_markdown_blocks       
)
from config import (
    H1_CONTENT_LIMIT,
    KEYWORD_SENTENCE_LIMIT,
    MAX_CONTENT_LENGTH,
    NUM_KEYWORDS,
    BATCH_SIZE
)

def is_homepage_url(url: str) -> bool:
    """
    Checks if a URL is likely to be a homepage based on its structure.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")  # Remove leading/trailing slashes

    # Check for common homepage patterns:
    if not path:  # Empty path (e.g., "https://www.example.com/")
        return True
    if path.lower() in ["home", "index.html", "index.php"]:
        return True
    if parsed_url.netloc.startswith("www.") and not path:
        return True

    return False


async def async_analyze_page(page_content: str, page_url: str, llm: ChatOpenAI) -> PageSegmentation:
    """
    Sends a snippet of text (Markdown) + URL to the LLM for classification,
    expecting a structured JSON response matching PageSegmentation.
    """
    parser = PydanticOutputParser(pydantic_object=PageSegmentation)
    prompt = PromptTemplate(
        template="""You are an SEO and content analysis assistant. Your goal is to classify webpages based on their content and purpose.

The text you are receiving **is not the full page content**. Instead, it consists of **selected snippets** extracted from key sections of the page:
- **Headings (H1, H2, H3)** and the content beneath them.
- **Keyword-relevant sentences** extracted using TF-IDF.
- The first **few paragraphs** when available.

Because this is a **partial representation of the page**, you should:
- Use the **provided URL** as additional context when needed.
- Focus on **overall page intent** rather than missing details.
- If the combined URL and extracted text lacks enough context to determine page categorizations, return **'Unable to determine'**.

Analyze the provided content and return JSON following the schema:

{schema}

**URL**: {page_url}

**Extracted Page Content (Partial)**:
{page_content}
""",
        input_variables=["page_content", "page_url"],
        partial_variables={"schema": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    try:
        result = await chain.ainvoke({"page_content": page_content, "page_url": page_url})
        return result
    except (OutputParserException, ValidationError) as e:
        logger.error(f"Parsing error for {page_url}: {e}")
        return PageSegmentation(
            page_url=page_url,
            page_type_l1="Error - Unable to Process Page",
            page_type_l2=None,
            industry=None,
            page_topic=None
        )
    except Exception as e:
        logger.exception(f"Unexpected error during analysis of {page_url}: {e}")
        return PageSegmentation(
            page_url=page_url,
            page_type_l1="Error - Unable to Process Page",
            page_type_l2=None,
            industry=None,
            page_topic=None
        )


async def process_one_url(url: str, llm: ChatOpenAI) -> PageSegmentation:
    """
    Fetch raw HTML & markdown for a page, run heading-based extraction (plus fallback),
    and send a final snippet of content to the LLM for classification.
    """
    # 1) URL-Based Homepage Check
    if is_homepage_url(url):
        # Return a quick label for a homepage
        return PageSegmentation(
            page_url=url,
            page_type_l1="Homepage",
            page_type_l2=None,
            industry=None,
            page_topic=None
        )

    # 2) Attempt to scrape the page
    try:
        resp = await async_scrape_url(url)
        raw_html = resp.get("html", "")
        md_fallback = resp.get("markdown", "")  # Old fallback

        # If no raw HTML was returned, we can't do heading-based extraction
        if not raw_html:
            logger.warning(f"No raw HTML for {url}. Attempting fallback snippet approach.")
            # If no markdown either, just do a minimal URL-based analysis:
            if not md_fallback:
                logger.warning(f"No content at all for {url}, analyzing by URL only.")
                return await async_analyze_page("No content. Use URL only.", url, llm)
            # Otherwise, fallback to the old snippet approach:
            return await fallback_snippet_approach(md_fallback, url, llm)

        # 3) Heading-based Extraction
        heading_blocks = extract_heading_blocks(raw_html, heading_tags=('h1', 'h2', 'h3'))
        if not heading_blocks:
            logger.warning(f"No heading blocks found for {url}. Using fallback snippet approach.")
            if md_fallback:
                return await fallback_snippet_approach(md_fallback, url, llm)
            else:
                logger.warning(f"No markdown fallback either. URL only for {url}.")
                return await async_analyze_page("No content. Use URL only.", url, llm)

        # 4) Convert each heading block to Markdown
        markdown_blocks = convert_blocks_to_markdown(heading_blocks)

        # 5) Chunk the blocks if they are large
        chunked_blocks = chunk_markdown_blocks(markdown_blocks, max_chars=MAX_CONTENT_LENGTH)

        # 6) Combine into a single string for analysis (watch out for total length)
        combined_md = "\n\n".join(chunked_blocks)
        if len(combined_md) > MAX_CONTENT_LENGTH:
            combined_md = combined_md[:MAX_CONTENT_LENGTH]

        # 7) Send to LLM for classification
        segmentation = await async_analyze_page(combined_md, url, llm)
        return segmentation

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return PageSegmentation(
            page_url=url,
            page_type_l1="Error - Unable to Process Page",
            page_type_l2=None,
            industry=None,
            page_topic=None
        )


async def fallback_snippet_approach(md_text: str, url: str, llm: ChatOpenAI) -> PageSegmentation:
    """
    The original snippet approach that:
      - extracts up to H1_CONTENT_LIMIT after H1
      - picks TF-IDF-based keywords
      - merges them, truncates, sends to LLM
    """
    if not md_text:
        # No content at all
        return await async_analyze_page("No content. Use URL only.", url, llm)

    # 1) Extract up to H1_CONTENT_LIMIT after the first H1
    h1_content = ""
    h1_start = -1
    if "<h1>" in md_text:
        h1_start = md_text.find("<h1>") + len("<h1>")
        h1_end = md_text.find("</h1>", h1_start)
    # Check for Markdown # at the start of a line
    if h1_start == -1:
        lines = md_text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("# "):
                h1_start = len("\n".join(lines[:i+1]))  # +1 for newline
                break

    if h1_start != -1:
        if 'h1_end' in locals():
            h1_content = md_text[h1_start : h1_end + H1_CONTENT_LIMIT]
        else:
            h1_content = md_text[h1_start : h1_start + H1_CONTENT_LIMIT]

    if not h1_content:
        h1_content = md_text[:H1_CONTENT_LIMIT]

    # 2) Simple TF-IDF for up to NUM_KEYWORDS
    keyword_sentences = []
    if len(md_text) > 0:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=NUM_KEYWORDS)
            vectorizer.fit([md_text])
            keywords = vectorizer.get_feature_names_out()

            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', md_text)
            for sentence in sentences:
                if any(k in sentence.lower() for k in keywords):
                    keyword_sentences.append(sentence)
                    if len(keyword_sentences) >= KEYWORD_SENTENCE_LIMIT:
                        break
        except ValueError as e:
            logger.warning(f"TF-IDF error on fallback for {url}: {e}. Using only H1 content.")
            keyword_sentences = []

    combined_content = h1_content + "\n".join(keyword_sentences)
    limited_content = combined_content[:MAX_CONTENT_LENGTH]

    return await async_analyze_page(limited_content, url, llm)


async def process_urls(state: FlowState, llm: ChatOpenAI) -> tuple[List[PageSegmentation], List[str]]:
    """
    Processes all URLs in batches of size BATCH_SIZE.
    Returns a tuple of (list_of_PageSegmentation, list_of_failed_urls).
    """
    urls = state["urls"]

    results: List[PageSegmentation] = []
    failed_urls: List[str] = []

    for i in range(0, len(urls), BATCH_SIZE):
        chunk = urls[i : i + BATCH_SIZE]
        logger.info(f"Processing chunk of {len(chunk)} URLs (index {i} to {i + BATCH_SIZE - 1})...")

        # Run all tasks in parallel
        tasks = [process_one_url(u, llm) for u in chunk]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions or valid results
        for url, result in zip(chunk, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {url} due to exception: {result}")
                failed_urls.append(url)
            else:
                results.append(result)

    return results, failed_urls
