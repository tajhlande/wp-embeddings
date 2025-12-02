import logging
import os
import sqlite3

from typing import Optional

import chromadb.utils.embedding_functions as embedding_functions

from chromadb.api.types import EmbeddingFunction, Embeddings
import openai

from classes import Page
from database import (
    get_page_by_id,
    get_page_ids_needing_embedding_for_chunk,
    get_sql_conn,
    upsert_embeddings_in_batch,
)
from progress_utils import ProgressTracker

# Configure logging to suppress INFO messages from httpx
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress INFO level logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_embedding_function(
    model_name: str,
    openai_compatible_url: str,
    openai_api_key: str,
    openai_api_version: str = "v1",
) -> EmbeddingFunction:
    # we're using OpenAI embedding function to point to an OpenAI compatible local server
    # because the JinaAI embedding function doesn't allow us to select our own server
    return embedding_functions.OpenAIEmbeddingFunction(
        # api_type='OpenAI',
        api_key=openai_api_key,
        api_base=openai_compatible_url,
        api_version=openai_api_version,
        model_name=model_name,
    )


def compute_page_embeddings(
    page: Page, embedding_function: EmbeddingFunction
) -> Embeddings:
    logger.debug("Computing embedding for page(%d) titled %s", page.page_id, page.title)
    text_content = f"{page.title}\n{page.abstract}"
    try:
        return embedding_function(text_content)
    except openai.InternalServerError as e:
        logger.warning(f"Caught openai.InternalServerError {e}")
        times_halved = 0
        while ("input is too large to process" in e.message or
               "input is too large to process" in str(e)) and len(text_content) > 0:
            try:
                logger.warning("Cutting text content in half to fit context for page %d "
                               "and retrying embedding", page.page_id)
                text_content = text_content[:len(text_content)//2]
                times_halved += 1
                result = embedding_function(text_content)
                logger.info("Successful embedding smaller content")
                return result
            except openai.InternalServerError as e2:
                e = e2

        if len(text_content) == 0:
            raise ValueError(f"Can't embed empty content for page {page.page_id}. "
                             f"Times halved: {times_halved}. Current text content: '{text_content}'")
        logger.warning(f"Some other kind of message besides 'input is too large to process': {e.message}")
        raise


def compute_embeddings_for_chunk(
    namespace: str,
    chunk_name: str,
    embedding_function: EmbeddingFunction,
    sqlconn: sqlite3.Connection,
    limit: Optional[int] = None,
    tracker: Optional[ProgressTracker] = None,
) -> None:

    page_id_list = get_page_ids_needing_embedding_for_chunk(chunk_name, sqlconn, namespace=namespace)

    # Apply limit if specified
    if limit and limit < len(page_id_list):
        page_id_list = page_id_list[:limit]

    tracker.set_total(len(page_id_list)) if tracker else None

    BUFFER_SIZE = 100
    buffer = []
    for page_id in page_id_list:
        page = get_page_by_id(namespace, page_id, sqlconn)
        if not page:
            logger.warning("Page with page_id %d not found, skipping.", page_id)
            continue

        # Assuming each line is a JSON object representing a page
        try:
            embeddings = compute_page_embeddings(page, embedding_function)
            embedding = embeddings[0]
            buffer.append((page.page_id, embedding))
            embedding = embeddings[0]
            if len(buffer) >= BUFFER_SIZE:
                upsert_embeddings_in_batch(namespace, buffer, sqlconn, BUFFER_SIZE)
                tracker.update(len(buffer)) if tracker else None
                buffer = []
        except Exception:
            logger.exception(f"Exception while computing embedding for page {page.page_id}")

    # flush remainder
    if buffer:
        upsert_embeddings_in_batch(namespace, buffer, sqlconn, BUFFER_SIZE)
        tracker.update(len(buffer)) if tracker else None


EMBEDDING_MODEL_NAME_KEY = "EMBEDDING_MODEL_NAME"
EMBEDDING_MODEL_API_URL_KEY = "EMBEDDING_MODEL_API_URL"
EMBEDDING_MODEL_API_KEY_KEY = "EMBEDDING_MODEL_API_KEY"

DEFAULT_EMBEDDING_MODEL_NAME = "jina-embeddings-v4-text-matching-GGUF"


def get_embedding_model_config() -> tuple[str, str, str]:
    """
    Retrieve embedding model configuration from environment variables.
    Returns: A tuple containing the embedding model name, API URL, and API key.
    Raises: ValueError if required environment variables are missing.
    """
    missing_env_vars = []
    embedding_model_name = os.environ.get(EMBEDDING_MODEL_NAME_KEY)
    if not embedding_model_name:
        embedding_model_name = DEFAULT_EMBEDDING_MODEL_NAME
        logger.warning(
            "%s environment variable not set, using default value: %s",
            EMBEDDING_MODEL_NAME_KEY,
            embedding_model_name,
        )
    if not os.environ.get(EMBEDDING_MODEL_API_URL_KEY):
        missing_env_vars.append(EMBEDDING_MODEL_API_URL_KEY)
    if not os.environ.get(EMBEDDING_MODEL_API_KEY_KEY):
        missing_env_vars.append(EMBEDDING_MODEL_API_KEY_KEY)
    if missing_env_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}\n"
            f"Add them to .env or to enviroment by other means."
        )
    embedding_model_api_url = os.environ[EMBEDDING_MODEL_API_URL_KEY]
    embedding_model_api_key = os.environ[EMBEDDING_MODEL_API_KEY_KEY]
    return embedding_model_name, embedding_model_api_url, embedding_model_api_key


if __name__ == "__main__":
    enwiki_ns_0 = "enwiki_namespace_0"
    enwiki_chunk_0 = "enwiki_namespace_0_chunk_0"

    sqlconn = get_sql_conn(enwiki_ns_0)

    embedding_model_name, embedding_model_api_url, embedding_model_api_key = (
        get_embedding_model_config()
    )

    jina_text_matching_function = get_embedding_function(
        model_name=embedding_model_name,
        openai_compatible_url=embedding_model_api_url,
        openai_api_key=embedding_model_api_key,
    )

    compute_embeddings_for_chunk(
        namespace=enwiki_ns_0,
        chunk_name=enwiki_chunk_0,
        embedding_function=jina_text_matching_function,
        sqlconn=sqlconn,
    )
