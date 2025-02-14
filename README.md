# Webpage Segmentation and Categorization with LangGraph (With Conditional Evaluation)

This project implements a robust and configurable system for automatically segmenting and categorizing webpages using LangChain, LangGraph, and OpenAI's GPT models. It takes a list of URLs as input, scrapes the content of each page, analyzes the content using an LLM, and then normalizes the extracted categories (industry and topic) to create consistent groupings for data analysis.

## Features

- **Web Scraping:** Scrapes webpage content and converts it to clean Markdown. Includes logic to extract content after an H1 heading (HTML or Markdown) and sentences containing relevant keywords.
- **AI-Powered Analysis:** Uses GPT models to analyze page content and extract key information:
  - Page Type (e.g., Commercial, Editorial, Navigational) - Two levels of granularity.
  - Page Intent (e.g., Informational, Commercial, Transactional) - Two levels of granularity.
  - Industry (domain-level)
  - Page Topic (page-level)
  - Publication/Update Date
- **Hierarchical Normalization:** Normalizes extracted categories (industry and topic) into consistent groups. Page topics are normalized _within_ their respective industries, providing more meaningful segmentation.
- **Normalization Evaluation and Retry:** Evaluates the quality of the normalization using an LLM and automatically retries the normalization process (up to a configurable limit) if the results are not satisfactory. Feedback from the evaluation is used to improve subsequent normalization attempts.
- **Configurable:** Key parameters (batch size, retry attempts, LLM model, temperature, input CSV path, content extraction limits) are configurable via a `config.ini` file.
- **Error Handling:** Robust error handling for web scraping, LLM calls, and parsing errors. Failed URLs are tracked.
- **Output:** Generates both a CSV file and a Markdown file containing the analysis results.
- **Token Usage Tracking:** Tracks and reports OpenAI API token usage and cost.
- **Modular Design:** Code is organized into separate modules for models, configuration, normalization logic, scraping, and the main workflow, making it easy to understand and maintain.

## File Structure

```
project_root/
├── config.ini          # Configuration file
├── main.py             # Main script, LangGraph workflow
├── models.py           # Pydantic data models
├── normalization.py    # Normalization logic (industry, topic)
├── scraping.py         # Web scraping and content extraction
├── utils.py            # Utility functions (scraping, CSV loading)
├── output/             # Output directory (created automatically)
│   ├── output.csv      # CSV output file
│   └── output.md       # Markdown output file (optional)
└── README.md           # This file
```

## Requirements

- Python 3.11+
- Required Python packages (install using `pip` from `requirements.txt`):
- Create an `.env` with the below variables

```
OPENAI_API_KEY=[your-key]
PROXY_HOST=[optional]
PROXY_USERNAME=[optional]
PROXY_PASSWORD=[optional]
```

- A CSV file (`input.csv` by default, configurable) containing a list of URLs to process. The CSV should have a header row, and the URLs should be in the first column. Example:

  ```csv
  url
  https://www.example.com/page1
  https://www.example.com/page2
  https://www.anotherdomain.com/blog/post1
  ```

## Setup

1. **Download Requirements & Add `.env` file**

2. **Update `config.ini` as needed:**

   Create a `config.ini` file in the project root with the following structure:

   ```ini
   [DEFAULT]
   batch_size = 50
   retry_attempts = 3
   min_pages_for_normalization = 10
   llm_model = gpt-4o
   temperature = 0.0
   input_csv_path = input.csv
   h1_content_limit = 500
   keyword_sentence_limit = 5
   max_content_length = 2000
   num_keywords = 10
   ```

   - `batch_size`: The number of URLs to process in each batch (to avoid rate limits).
   - `retry_attempts`: The maximum number of times to retry normalization if the evaluation fails.
   - `min_pages_for_normalization`: The minimum number of pages required before normalization is performed.
   - `llm_model`: The OpenAI model to use (e.g., `gpt-4o`, `gpt-3.5-turbo`). `gpt-4o` is recommended.
   - `temperature`: The temperature setting for the LLM (controls randomness). 0.0 is recommended for consistent results.
   - `input_csv_path`: The path to your input CSV file.
   - `h1_content_limit`: The maximum number of characters to extract after an H1 heading.
   - `keyword_sentence_limit`: The maximum number of sentences containing keywords to extract.
   - `max_content_length`: The maximum total length (in characters) of the content to send to the LLM for analysis.
   - `num_keywords`: The number of keywords to extract for keyword-based sentence selection.

3. **Create `input.csv`:**

   Create a CSV file (e.g., `input.csv`) with a list of URLs, one URL per row. See the example above.

## Usage

1.  **Run the script:**

    ```bash
    python main.py
    ```

2.  **Output:**

    - The script will create an `output` directory (if it doesn't exist).
    - A CSV file (`output.csv`) containing the analysis results will be written to the `output` directory.
    - A Markdown file (`output.md`) containing a human-readable summary of the results will also be written to the `output` directory.
    - Any URLs that consistently failed to be processed will be logged to the console.

## Code Overview

- **`main.py`:** The main entry point. Defines the LangGraph workflow, loads URLs, runs the workflow, and saves the output.
- **`models.py`:** Defines the Pydantic data models (e.g., `PageSegmentation`, `NormalizedMapping`).
- **`normalization.py`:** Contains functions for normalizing industry and page topic values.
- **`scraping.py`:** Handles web scraping and content extraction (including H1 and keyword-based extraction).
- **`config.py`:** Loads and provides access to configuration parameters.
- **`utils.py`:** Contains utility functions (e.g., `async_scrape_url`, `load_urls_from_csv`).

## Troubleshooting

- **`KeyError: 'mappings'`:** This usually means the prompt template in `normalization.py` is incorrect. Ensure the placeholders in the template match the keys you're using in the `.format()` call.
- **`OutputParserException` / `ValidationError`:** These errors mean the LLM output doesn't match the expected Pydantic model. Check the prompts and ensure they are clear and provide examples.
- **`AttributeError: 'AIMessage' object has no attribute 'strip'`:** You're trying to call `.strip()` on an LLM message object. Use `.content.strip()` to access the text content first.
- **Slow Performance:** Processing many URLs can take time. Adjust `batch_size` in `config.ini`.
- **Missing Imports:** Ensure all required libraries are installed (see Requirements). Make sure you've imported necessary modules within the files where they are used (e.g., `TfidfVectorizer` needs to be imported where it's used).
- **Timeout Errors:** If you encounter timeout errors during scraping, you might need to add retry logic and error handling to your `async_scrape_url` function in `utils.py`. You could also consider using a more robust scraping library like `Scrapy`.
