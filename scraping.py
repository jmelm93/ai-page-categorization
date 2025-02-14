import re
import asyncio  
from typing import List
from loguru import logger
from pydantic import ValidationError
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from models import PageSegmentation, FlowState
from utils import async_scrape_url
from config import H1_CONTENT_LIMIT, KEYWORD_SENTENCE_LIMIT, MAX_CONTENT_LENGTH, NUM_KEYWORDS, BATCH_SIZE

from sklearn.feature_extraction.text import TfidfVectorizer

def is_homepage_url(url: str) -> bool:
    """
    Checks if a URL is likely to be a homepage based on its structure.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")  # Remove leading/trailing slashes

    # Check for common homepage patterns:
    if not path:  # Empty path (e.g., "https://www.example.com/")
        return True
    if path == "home" or path == "index.html" or path == "index.php": #common homepage paths
        return True
    if parsed_url.netloc.startswith("www."): #check if www. is in URL and path is empty
        if not path:
            return True
    return False


async def async_analyze_page(page_content: str, page_url: str, llm: ChatOpenAI) -> PageSegmentation:
    parser = PydanticOutputParser(pydantic_object=PageSegmentation)
    prompt = PromptTemplate(
        template="""You are an SEO and content analysis assistant. Analyze the markdown content and return JSON. 
If no content is available, use the URL for analysis. If URL doesn't contain sufficient context to group, return 'Unable to determine'.

{schema}

**URL**: {page_url}
**Content (contains part of page content)**: 
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
        return PageSegmentation(page_url=page_url, page_type_l1="Error - Unable to Process Page", page_type_l2=None, industry=None, page_topic=None)
    except Exception as e:
        logger.exception(f"Unexpected error during analysis of {page_url}: {e}")
        return PageSegmentation(page_url=page_url, page_type_l1="Error - Unable to Process Page", page_type_l2=None, industry=None, page_topic=None)


async def process_one_url(url: str, llm: ChatOpenAI) -> PageSegmentation:
    # --- URL-Based Homepage Check FIRST ---
    if is_homepage_url(url):
        return PageSegmentation(page_url=url, page_type_l1="Homepage", page_type_l2=None, industry=None, page_topic=None)
    
    # --- If not a homepage, proceed with content extraction ---
    try:
        resp = await async_scrape_url(url)
        md_text = resp.get("markdown", "")
        if not md_text:
            logger.warning(f"No text for {url}. Running analysis with URL only.")
            segmentation = await async_analyze_page("No content extraction available for this page. Use solely the URL for analysis.", url, llm)
            return segmentation

        # --- Content Extraction Logic ---
        h1_content = ""
        keyword_sentences = []

        # 1. Extract content after H1 (HTML or Markdown)
        h1_start = -1
        # Check for HTML <h1> tag
        if "<h1>" in md_text:
            h1_start = md_text.find("<h1>") + len("<h1>")
            h1_end = md_text.find("</h1>", h1_start)
        # Check for Markdown H1 (# at the beginning of a line)
        if h1_start == -1:
            lines = md_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    h1_start = len("\n".join(lines[:i+1])) # +1 to include the newline
                    break  # Only find the *first* H1

        if h1_start != -1:
            if 'h1_end' in locals(): #checks if h1_end is defined
                h1_content = md_text[h1_start:h1_end + H1_CONTENT_LIMIT]
            else: #if not defined, just go with h1_start
                h1_content = md_text[h1_start: h1_start + H1_CONTENT_LIMIT]

        # If no H1, use the beginning of the content
        if not h1_content:
            h1_content = md_text[:H1_CONTENT_LIMIT]


        # 2. Keyword Extraction and Sentence Selection
        if len(md_text) > 0:  # Avoid errors with empty content
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=NUM_KEYWORDS) #remove common english words
                vectorizer.fit([md_text])
                keywords = vectorizer.get_feature_names_out()

                # Split into sentences (basic sentence splitting)
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', md_text)

                # Select sentences containing keywords
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        keyword_sentences.append(sentence)
                        if len(keyword_sentences) >= KEYWORD_SENTENCE_LIMIT:
                            break
            except ValueError as e:
                logger.warning(f"TF-IDF or sentence splitting error on {url}: {e}.  Using h1_content only.")
                keyword_sentences = [] # Empty the list, as an error occurred.

        # 3. Combine and Limit Content
        combined_content = h1_content + "\n".join(keyword_sentences)
        limited_content = combined_content[:MAX_CONTENT_LENGTH]
        
        segmentation = await async_analyze_page(limited_content, url, llm) # Pass limited content and llm
        return segmentation

    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return PageSegmentation(page_url=url, page_type_l1="Error - Unable to Process Page", page_type_l2=None, industry=None, page_topic=None)

async def process_urls(state: FlowState, llm: ChatOpenAI) -> tuple[List[PageSegmentation], List[str]]:
    urls = state["urls"]
    
    results: List[PageSegmentation] = []
    failed_urls: List[str] = []

    for i in range(0, len(urls), BATCH_SIZE):
        chunk = urls[i : i + BATCH_SIZE]
        logger.info(f"Processing chunk of {len(chunk)} URLs (from index {i} to {i+BATCH_SIZE-1})...")
        tasks = [process_one_url(u, llm) for u in chunk]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(chunk, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {url}: {result}")
                failed_urls.append(url)
            else:
                results.append(result)

    return results, failed_urls