import asyncio
import os
import io
import csv as py_csv
import pandas as pd
import datetime
from typing import List
from loguru import logger

# LangChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import OpenAICallbackHandler

# LangGraph
from langgraph.graph import END, START, StateGraph

# Local imports
from utils import load_urls_from_csv
from config import LLM_MODEL_ADVANCED, LLM_MODEL_FAST, TEMPERATURE, INPUT_CSV_PATH, RETRY_ATTEMPTS
from models import FlowState
import normalization
import scraping

# --- LLM Setup ---
callback_handler = OpenAICallbackHandler()
llm_advanced = ChatOpenAI(model=LLM_MODEL_ADVANCED, temperature=TEMPERATURE, callbacks=[callback_handler])
llm_fast = ChatOpenAI(model=LLM_MODEL_FAST, temperature=TEMPERATURE, callbacks=[callback_handler])

# --- Node Functions ---

async def process_urls_node(state: FlowState) -> dict:
    logger.info("Processing URLs...")
    results, failed_urls = await scraping.process_urls(state, llm=llm_fast)
    return {"results": results, "failed_urls": failed_urls} # Update state keys

async def normalize_industry_node(state: FlowState) -> dict:
    logger.info("Normalizing industry...")
    updated_results = await normalization.normalize_industry(state, llm=llm_advanced)
    return {"results": updated_results} # Update state keys

async def normalize_page_topic_node(state: FlowState) -> dict:
    logger.info("Normalizing page topic...")
    updated_results = await normalization.normalize_page_topic(state, llm=llm_advanced)
    return {"results": updated_results} # Update state keys

async def evaluate_normalization_node(state: FlowState) -> dict:
    logger.info("Evaluating normalization...")
    result = await normalization.evaluate_normalization(state, llm=llm_advanced)
    return result # No unpacking needed to update state keys


def should_retry(state: FlowState) -> str:
    logger.info("Checking if normalization should be retried...")
    # Correctly check eval_status
    if state["retry_count"] < RETRY_ATTEMPTS and state.get("eval_status") == "failure":
        logger.info("Retrying normalization...")
        return "retry"
    else:
        logger.info("Normalization complete.")
        return "aggregate"


# --- Aggregation and Output ---
def aggregator(state: FlowState) -> dict:
    logger.info("Aggregating results...")
    
    results = state["results"]
    failed_urls = state["failed_urls"]

    all_md = []
    for idx, seg in enumerate(results, start=1):
        md_part = f"## Page {idx} for URL: {seg.page_url}\n\n{seg.to_markdown}"
        all_md.append(md_part)
    final_markdown = "\n\n---\n\n".join(all_md)

    data = []
    for seg in results:
        data.append(
            {
                "page_url": seg.page_url or "",
                "page_type_l1": seg.page_type_l1 or "",
                "page_type_l2": seg.page_type_l2 or "",
                "industry": seg.industry or "",
                "page_topic": seg.page_topic or "",
                "industry_normalized": seg.industry_normalized or "",
                "page_topic_normalized": seg.page_topic_normalized or "",
            }
        )
    df = pd.DataFrame(
        data,
        columns=[
            "page_url",
            "page_type_l1",
            "page_type_l2",
            "industry",
            "page_topic",
            "industry_normalized",
            "page_topic_normalized",
        ],
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, quoting=py_csv.QUOTE_ALL)
    final_csv = buffer.getvalue()

    return {
        "final_markdown": final_markdown,
        "final_csv": final_csv,
        "failed_urls": failed_urls,
    }


# --- Graph Definition ---
workflow = StateGraph(FlowState)
workflow.add_node("process_urls", process_urls_node)  # Use the node functions
workflow.add_node("normalize_industry", normalize_industry_node)
workflow.add_node("normalize_page_topic", normalize_page_topic_node)
workflow.add_node("evaluate_normalization", evaluate_normalization_node)
workflow.add_node("aggregator", aggregator)

workflow.add_edge(START, "process_urls")
workflow.add_edge("process_urls", "normalize_industry")
workflow.add_edge("normalize_industry", "normalize_page_topic")
workflow.add_edge("normalize_page_topic", "evaluate_normalization")

workflow.add_conditional_edges(
    "evaluate_normalization",
    should_retry,
    {"retry": "normalize_industry", "aggregate": "aggregator"},
)
workflow.add_edge("aggregator", END)

async_app = workflow.compile()


# --- Main Function ---
async def run_flow_async(urls: List[str]):
     # Create LLM here and add to initial state
    initial_state = {
        "urls": urls,
        "results": [],
        "final_markdown": "",
        "final_csv": "",
        "retry_count": 0,
        "evaluation_notes": "",
        "failed_urls": [],
    }
    final_state = await async_app.ainvoke(initial_state)
    return final_state

async def main():
    """Main function to run the workflow."""
    urls = load_urls_from_csv(INPUT_CSV_PATH)
    if not urls:
        logger.error(f"No URLs found in {INPUT_CSV_PATH}, exiting.")
        return

    logger.info(f"Loaded {len(urls)} URLs from {INPUT_CSV_PATH}.")

    # with get_openai_callback() as cb:
    final_state = await run_flow_async(urls)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"page_groups_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    csv_filename = os.path.join(output_dir, f"{filename}.csv")
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write(final_state["final_csv"])

    md_filename = os.path.join(output_dir, f"{filename}.md")
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(final_state["final_markdown"])

    logger.info(f"Processing complete.  CSV written to {csv_filename}")
    if final_state["failed_urls"]:
        logger.warning(f"Failed URLs: {final_state['failed_urls']}")

    # log_openai_callback_costs("main", callback_handler)
    logger.info(f"Total Tokens: {callback_handler.total_tokens}")
    logger.info(f"Prompt Tokens: {callback_handler.prompt_tokens}")
    logger.info(f"Completion Tokens: {callback_handler.completion_tokens}")
    logger.info(f"Total Cost (USD): ${callback_handler.total_cost}")

if __name__ == "__main__":
    asyncio.run(main())