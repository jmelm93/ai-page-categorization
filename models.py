import datetime
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# --- Models (Structured Output ---
class PageSegmentation(BaseModel):
    """
    Represents a structured segmentation of a webpage's content.
    """
    page_url: str = Field(..., description="The full URL of the webpage being analyzed.")
    page_type_l1: str = Field(..., description="""High-level webpage type:
    - 'Commercial': Selling products/services (e.g., product pages).
    - 'Editorial': Content-driven (e.g., blog posts, news).
    - 'Navigational': Help users find content (e.g., homepages).
    - 'Transactional': User completes an action (e.g., contact forms).
    - 'Community': User interaction (e.g., forums).
    - 'Other': If none of the above fit.
    """)
    page_type_l2: Optional[str] = Field(None, description="""Specific webpage sub-type:
    - If 'Commercial': 'Product Listing', 'Product Detail', 'Pricing'.
    - If 'Editorial': 'Blog Post', 'News Article', 'Review', 'Guide'.
    - If 'Navigational': 'Homepage', 'Category Page', 'About Us'.
    - If 'Transactional': 'Contact Form', 'Registration Form'.
    - If 'Community': 'Forum', 'Thread', 'User Profile'.
    """)
    page_intent_l1: str = Field(..., description="""High-level user intent:
    - 'Informational': Seeking information.
    - 'Commercial': Researching products/services.
    - 'Transactional': Completing an action.
    - 'Navigational': Finding a specific page.
    """)
    page_intent_l2: Optional[str] = Field(None, description="""Specific user intent:
    - If 'Informational': 'Learn', 'Research', 'Compare'.
    - If 'Commercial': 'Browse', 'Evaluate', 'Read Reviews'.
    - If 'Transactional': 'Purchase', 'Sign Up', 'Contact'.
    - If 'Navigational': 'Find Location', 'Access Account'.
    """)
    industry: Optional[str] = Field(None, description="""The website's (domain's) primary industry (broad).
    E.g., 'Technology', 'Finance', 'Healthcare', 'Retail'. Be specific within these (e.g., 'Consumer Electronics').""")
    page_topic: Optional[str] = Field(None, description="""The specific subject matter or topic of *this* page (granular).
    E.g., if industry is 'Technology', topic might be 'Cloud Storage Solutions' or 'iPhone 15 Review'.""")
    extracted_date: Optional[datetime.date] = Field(None, description="Publication/update date from page content.")

    # Normalized fields
    industry_normalized: Optional[str] = None
    page_topic_normalized: Optional[str] = None

    @property
    def to_markdown(self) -> str:
        md = f"# Page Type (L1): {self.page_type_l1}\n"
        if self.page_type_l2:
            md += f"- **Type (L2):** {self.page_type_l2}\n"
        md += f"- **Intent (L1):** {self.page_intent_l1}\n"
        if self.page_intent_l2:
            md += f"- **Intent (L2):** {self.page_intent_l2}\n"
        if self.industry:
            md += f"- **Domain Industry:** {self.industry}\n"
        if self.page_topic:
            md += f"- **Page Topic:** {self.page_topic}\n"
        if self.industry_normalized:
            md += f"- **Industry (Normalized):** {self.industry_normalized}\n"
        if self.page_topic_normalized:
            md += f"- **Topic (Normalized):** {self.page_topic_normalized}\n"
        if self.extracted_date:
            md += f"- **Date:** {self.extracted_date.strftime('%Y-%m-%d')}\n"
        return md


class NormalizedMapping(BaseModel):
    """
    Mapping from an original value to a normalized value.
    """
    original_value: str = Field(..., description="Original, non-standardized value.")
    normalized_value: str = Field(..., description="Standardized, normalized value.")


class NormalizedMappingResponse(BaseModel):
    """
    Response containing a list of normalized mappings.
    """
    mappings: List[NormalizedMapping] = Field(..., description="List of `NormalizedMapping` objects.")


class NormalizationEvaluation(BaseModel):
    """
    Result of evaluating the normalization quality.
    """
    status: str = Field(..., description="'success' or 'failure'.")
    notes: str = Field(..., description="Feedback if 'failure'; empty if 'success'.")
    

# --- Flow State ---
class FlowState(TypedDict):
    urls: List[str]
    results: List[PageSegmentation]
    final_markdown: str
    final_csv: str
    retry_count: int
    evaluation_notes: str
    failed_urls: List[str]