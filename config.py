import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Make the config values easily accessible as constants
BATCH_SIZE = config.getint("DEFAULT", "batch_size")
RETRY_ATTEMPTS = config.getint("DEFAULT", "retry_attempts")
# MIN_PAGES_FOR_NORMALIZATION = config.getint("DEFAULT", "min_pages_for_normalization")
LLM_MODEL_ADVANCED = config.get("DEFAULT", "llm_model_advanced")
LLM_MODEL_FAST = config.get("DEFAULT", "llm_model_fast")
TEMPERATURE = config.getfloat("DEFAULT", "temperature")
INPUT_CSV_PATH = config.get("DEFAULT", "input_csv_path")
H1_CONTENT_LIMIT = config.getint("DEFAULT", "h1_content_limit")
KEYWORD_SENTENCE_LIMIT = config.getint("DEFAULT", "keyword_sentence_limit")
MAX_CONTENT_LENGTH = config.getint("DEFAULT", "max_content_length")
NUM_KEYWORDS = config.getint("DEFAULT", "num_keywords")