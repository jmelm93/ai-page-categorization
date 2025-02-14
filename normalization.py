from typing import Dict, List
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI  

from models import NormalizedMappingResponse, PageSegmentation, NormalizationEvaluation, FlowState
from config import RETRY_ATTEMPTS  # MIN_PAGES_FOR_NORMALIZATION, 



async def get_structured_mappings(prompt_str: str, llm: ChatOpenAI) -> Dict[str, str]:
    parser = PydanticOutputParser(pydantic_object=NormalizedMappingResponse)
    prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])
    chain = prompt | llm | parser

    try:
        logger.info("Requesting structured mappings from AI.")
        result = await chain.ainvoke({"prompt": prompt_str})

        # Try to parse as NormalizedMappingResponse.  If it fails, try fallback parsing.
        try:
            final_map = {item.original_value.strip(): item.normalized_value.strip() for item in result.mappings}
        except AttributeError:  # result doesn't have .mappings
            logger.warning("Falling back to manual parsing of LLM response.")
            final_map = {}
            # Fallback parsing logic (attempt to extract mappings)
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, list):  # Handle list-based responses
                        for item in value:
                            if isinstance(item, dict) and "original" in item and "normalized" in item:
                                final_map[item["original"].strip()] = item["normalized"].strip()
                            elif isinstance(item, dict) and "original_value" in item and "normalized_value" in item:
                                final_map[item["original_value"].strip()] = item["normalized_value"].strip()
                    elif isinstance(value, str): #if string, assume single mapping
                        final_map[key.strip()] = value.strip()
            elif isinstance(result, str): # Handle string
                logger.warning(f"LLM Response was not a dict: {result}")
                #add single string response
                final_map[result] = result
        return final_map


    except Exception as e:
        logger.error(f"Error retrieving structured mappings: {e}")
        return {}


# async def normalize_industry(results: List[PageSegmentation], notes: str, llm: ChatOpenAI) -> List[PageSegmentation]:
async def normalize_industry(state: FlowState) -> List[PageSegmentation]:
    results = state["results"]
    notes = state["evaluation_notes"]
    llm = state["llm"]
    
    # if len(results) < MIN_PAGES_FOR_NORMALIZATION:
    #     logger.info("Skipping industry normalization (not enough pages).")
    #     return results

    distinct_vals = {
        (r.industry or "").strip() for r in results if r.industry
    }
    if not distinct_vals:
        logger.info("No industry values found.")
        return results

    prompt_str = f"""You are an expert data analyst tasked with creating a normalized set of categories.
Normalize the following industry values into a concise, consistent set of categories suitable for data analysis.
Group similar values together, and choose the *most specific* term that accurately represents each group.

Example:
Input: ['Software', 'Tech', 'Software Development', 'Tech - Hardware']
Output:
{{
  "mappings": [
    {{"original_value": "Software", "normalized_value": "Technology - Software"}},
    {{"original_value": "Tech", "normalized_value": "Technology - General"}},
    {{"original_value": "Software Development", "normalized_value": "Technology - Software"}},
    {{"original_value": "Tech - Hardware", "normalized_value": "Technology - Hardware"}}
  ]
}}

Input Values:
{list(distinct_vals)}

{notes}

Return JSON in the EXACT format shown in the example, with a top-level "mappings" key.
"""
    industry_map = await get_structured_mappings(prompt_str, llm)  # Pass llm
    for r in results:
        if r.industry:  # Check if r.industry is not None
            original_value = r.industry.strip()
            if original_value in industry_map:
                r.industry_normalized = industry_map[original_value]
            else:
                r.industry_normalized = None

    return results


# async def normalize_page_topic(results: List[PageSegmentation], notes: str, llm: ChatOpenAI) -> List[PageSegmentation]:
async def normalize_page_topic(state: FlowState) -> List[PageSegmentation]:
    results = state["results"]
    notes = state["evaluation_notes"]
    llm = state["llm"]

    # if len(results) < MIN_PAGES_FOR_NORMALIZATION:
    #     logger.info("Skipping page_topic normalization (not enough pages).")
    #     return results

    industry_grouped_topics = {}
    for r in results:
        if r.industry_normalized and r.page_topic:
            industry_grouped_topics.setdefault(r.industry_normalized, set()).add(r.page_topic.strip())

    for industry, topics in industry_grouped_topics.items():
        prompt_str = f"""You are an expert data analyst tasked with creating a normalized set of BROAD categories.
Normalize the following page topic values *within the context of the industry '{industry}'.
Create CONCISE, GENERAL categories suitable for data analysis across a wide range of pages.  
Group similar values together into BROADER categories.

Example (Industry: Technology):
Input: ['Cloud Storage Pricing', 'cloud storage', 'AWS S3', 'Notion', 'project management', 'software development', 'data engineering']
Output:
{{
  "mappings": [
    {{"original_value": "Cloud Storage Pricing", "normalized_value": "Cloud"}},
    {{"original_value": "cloud storage", "normalized_value": "Cloud"}},
    {{"original_value": "AWS S3", "normalized_value": "Cloud"}},
    {{"original_value": "Notion", "normalized_value": "Productivity"}},
    {{"original_value": "project management", "normalized_value": "Productivity"}},
    {{"original_value": "software development", "normalized_value": "Development"}},
    {{"original_value": "data engineering", "normalized_value": "Development"}}
  ]
}}

Example (Industry: Alcohol):
Input: ['white wine', 'red wine', 'chardonnay', 'cabernet sauvignon', 'grey goose', 'vodka', 'gin']
Output:
{{
  "mappings": [
    {{"original_value": "white wine", "normalized_value": "Wine"}},
    {{"original_value": "red wine", "normalized_value": "Wine"}},
    {{"original_value": "pairing wine with food", "normalized_value": "Wine"}},
    {{"original_value": "grey goose", "normalized_value": "Spirits"}},
    {{"original_value": "vodka", "normalized_value": "Spirits"}},
    {{"original_value": "gin", "normalized_value": "Spirits"}}
  ]
}}

Example (Industry: Personal Finance):
Input: ['How to Make Money with a Side Gig', 'How to Save Money on Groceries', 'Investing in Stocks', 'Credit Card Rewards']
Output:
{{
  "mappings": [
    {{"original_value": "How to Make Money with a Side Gig", "normalized_value": "Money"}},
    {{"original_value": "How to Save Money on Groceries", "normalized_value": "Money"}},
    {{"original_value": "Investing in the Stock Market", "normalized_value": "Investing"}},
    {{"original_value": "Credit Card Reward Projects", "normalized_value": "Credit Cards"}}
  ]
}}

Input Values:
{list(topics)}

{notes}

Return JSON in the EXACT format shown in the example, with a top-level "mappings" key.
"""
        topic_map = await get_structured_mappings(prompt_str,llm) # Pass llm
        for r in results:
            if r.industry_normalized == industry and r.page_topic:
                original_value = r.page_topic.strip()
                if original_value in topic_map:
                    r.page_topic_normalized = topic_map[original_value]
                else:
                    r.page_topic_normalized = None

    return results

async def evaluate_normalization(state) -> dict:
    results = state["results"]
    retry_count = state["retry_count"]
    llm = state["llm"]
    
    if retry_count >= RETRY_ATTEMPTS:
        logger.info("Max retries reached. Skipping evaluation.")
        return {"evaluation_notes": "", "retry_count": retry_count, "eval_status": "success"} # Return a dict

    eval_data = [
        {
            "industry": r.industry,
            "industry_normalized": r.industry_normalized,
            "page_topic": r.page_topic,
            "page_topic_normalized": r.page_topic_normalized,
        }
        for r in results
    ]

    parser = PydanticOutputParser(pydantic_object=NormalizationEvaluation)
    
    # print('eval_data:', eval_data)
    # print('parser.get_format_instructions()', parser.get_format_instructions())

    prompt = PromptTemplate(
        template="""Evaluate normalization for data segmentation. In order to do this, compare the "industry" to "industry_normalized" and "page_topic" to "page_topic_normalized".
Are normalized categories distinct, meaningful, granular? Do they effectively group similar values from the original data into the normalized categories? 
If so, return 'success'. 
If not, return 'failure' with feedback on how to improve the normalization.
**IMPORTANT**: Sometimes the original is already a normalized value. In these cases, the normalized value should be the same as the original. This scenario should not be considered a failure.
**Examples of Good Normalization**:
- Original: 'Walking Dogs', Normalized: 'Pets'
- Original: 'Tech', Normalized: 'Technology'
- Original: 'red wine', Normalized: 'Wine'
- Original: 'grey goose', Normalized: 'Spirits'
- Original: 'AWS S3', Normalized: 'Cloud'
- Original: 'How to Make Money with a Side Gig', Normalized: 'Money'
**Normalized Pairings to QA**: 
{eval_data}
**Output Instructions**: 
{format_instructions}""",
        input_variables=["eval_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    try:
        evaluation = await chain.ainvoke({"eval_data": eval_data})
        logger.info(f"Normalization evaluation: {evaluation.status}")
        if evaluation.status == "failure":
            retry_count += 1
            logger.info(f"Evaluation failed. Retry count: {retry_count}. Evaluation notes: {evaluation.notes}")
        return {"evaluation_notes": evaluation.notes, "retry_count": retry_count, "eval_status": evaluation.status}
    except Exception as e:
        logger.error(f"Error during normalization evaluation: {e}")
        return {"evaluation_notes": "Error during evaluation.", "retry_count": retry_count, "eval_status": "failure"}