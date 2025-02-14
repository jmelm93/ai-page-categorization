import datetime
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# --- Models (Structured Output ---
# class PageSegmentation(BaseModel):
#     """
#     Represents a structured segmentation of a webpage's content.
#     """
#     page_url: str = Field(..., description="The full URL of the webpage being analyzed.")
#     page_type_l1: str = Field(..., description="""High-level webpage type:
#     - 'Commercial': Selling products/services (e.g., product pages).
#     - 'Editorial': Content-driven (e.g., blog posts, news).
#     - 'Homepage': Main entry point.
#     - 'Brand': Company information (e.g., about us, contact).
#     - 'Community': User interaction (e.g., forums).
#     - 'Other': If none of the above fit.
#     """)
#     page_type_l2: Optional[str] = Field(None, description="""Specific webpage sub-type:
#     - If 'Commercial': 'Category (Product Listing)', 'Product Detail', 'Pricing', etc.
#     - If 'Editorial': 'Blog Post', 'News Article', 'Review', 'Guide', etc.
#     - If 'Homepage': return '-' (no sub-type).
#     - If 'Brand': 'Contact Form', 'Registration Form', 'About Us', etc.
#     - If 'Community': 'Forum', 'Thread', 'User Profile', etc.
#     """)
#     page_intent_l1: str = Field(..., description="""High-level user intent:
#     - 'Informational': Seeking information.
#     - 'Commercial': Researching products/services.
#     - 'Transactional': Completing an action.
#     - 'Navigational': Finding a specific page.
#     """)
#     page_intent_l2: Optional[str] = Field(None, description="""Specific user intent:
#     - If 'Informational': 'Learn', 'Research', 'Compare'.
#     - If 'Commercial': 'Browse', 'Evaluate', 'Read Reviews'.
#     - If 'Transactional': 'Purchase', 'Sign Up', 'Contact'.
#     - If 'Navigational': 'Find Location', 'Access Account'.
#     """)
#     industry: Optional[str] = Field(None, description="""The website's (domain's) primary industry (broad).
#     E.g., 'Technology', 'Finance', 'Healthcare', 'Retail'. Be specific within these (e.g., 'Consumer Electronics').""")
#     page_topic: Optional[str] = Field(None, description="""The specific subject matter or topic of *this* page (granular).
#     E.g., if industry is 'Technology', topic might be 'Cloud Storage Solutions' or 'iPhone 15 Review'.""")

#     # Normalized fields
#     industry_normalized: Optional[str] = None
#     page_topic_normalized: Optional[str] = None

#     @property
#     def to_markdown(self) -> str:
#         md = f"# Page Type (L1): {self.page_type_l1}\n"
#         if self.page_type_l2:
#             md += f"- **Type (L2):** {self.page_type_l2}\n"
#         md += f"- **Intent (L1):** {self.page_intent_l1}\n"
#         if self.page_intent_l2:
#             md += f"- **Intent (L2):** {self.page_intent_l2}\n"
#         if self.industry:
#             md += f"- **Domain Industry:** {self.industry}\n"
#         if self.page_topic:
#             md += f"- **Page Topic:** {self.page_topic}\n"
#         if self.industry_normalized:
#             md += f"- **Industry (Normalized):** {self.industry_normalized}\n"
#         if self.page_topic_normalized:
#             md += f"- **Topic (Normalized):** {self.page_topic_normalized}\n"
#         return md


class PageSegmentation(BaseModel):
    """
    Represents a structured segmentation of a webpage's content.
    """
    page_url: str = Field(..., description="The full URL of the webpage being analyzed.")
    page_type_l1: str = Field(..., description="""L1 Categories:
    - Homepage: The main entry point of the website. Whenever the url path is '/' (root). Sometimes may have a url path like '/home', '/index', etc.
    - Commercial: Pages primarily focused on selling products or services.
    - Informational: Pages primarily focused on providing information like articles, blog posts, guides etc.
    - Navigation: Pages focused on helping users find specific information.
    - Brand: Pages about the company itself.
    - Community: Pages focused on user interaction.
    - Other: Pages that don't fit into the above categories.
    """)
    page_type_l2: Optional[str] = Field(None, description="""L2 Categories (Examples - Not Exhaustive):
    - If 'Homepage': return '-' (no sub-type).
    - If 'Commercial': Category (Product Listing), Product Detail, Pricing, Affiliate Offers, etc.
    - If 'Informational': How To, Listicle, Review, Guide, Calculator, Comparison, etc.
    - If 'Navigation': Homepage, Store Locator, Site Map, Search Results, Category (Navigation), etc.
    - If 'Community': Forum, Thread, User Profile, Comments Section, etc.
    - If 'Brand': About Us, Contact Us, Careers, Press, etc.
    - If 'Other': Error Page, Miscellaneous, etc.
    """)
    # page_type_l3: Optional[str] = Field(None, description="""L3 Categories (Optional - Use only when necessary for further distinction).""")
    # industry: Optional[str] = Field(None, description="""The website's (domain's) primary industry (broad). E.g., 'Technology', 'Finance', 'Healthcare', 'Retail'. Be specific within these (e.g., 'Consumer Electronics').""")
    industry: Optional[str] = Field(None, description="""The industry of focus for the page. E.g., 'Technology', 'Finance', 'Healthcare', 'Retail'. Be specific within these (e.g., 'Consumer Electronics').""")
    page_topic: Optional[str] = Field(None, description="""The specific subject matter or topic of *this* page (granular). E.g., if industry is 'Technology', topic might be 'Cloud Storage Solutions' or 'iPhone 15 Review'.""")

    # Normalized fields
    industry_normalized: Optional[str] = None
    page_topic_normalized: Optional[str] = None

    @property
    def to_markdown(self) -> str:
        md = f"- **Page Type (L1)**: {self.page_type_l1}\n"
        if self.page_type_l2:
            md += f"- **Sub Type (L2):** {self.page_type_l2}\n"
        if self.industry:
            md += f"- **Domain Industry:** {self.industry}\n"
        if self.page_topic:
            md += f"- **Page Topic:** {self.page_topic}\n"
        if self.industry_normalized:
            md += f"- **Industry (Normalized):** {self.industry_normalized}\n"
        if self.page_topic_normalized:
            md += f"- **Topic (Normalized):** {self.page_topic_normalized}\n"
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