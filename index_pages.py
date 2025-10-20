import json
import logging
import sqlite3
import time

from contextlib import contextmanager

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from chromadb.api.types import EmbeddingFunction, Embeddings
from chromadb.api.models.Collection import Collection as ChromaCollection

from common import get_any_page, get_page_by_id, get_page_ids_needing_embedding_for_chunk, get_sql_conn, update_embeddings_for_page

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_embedding_function(model_name: str,
                  openai_compatible_url: str,
                  openai_api_key: str,
                  openai_api_version: str = 'v1') -> EmbeddingFunction:
    # we're using OpenAI embedding function to point to an OpenAI compatible local server
    # because the JinaAI embedding function doesn't allow us to select our own server
    return embedding_functions.OpenAIEmbeddingFunction(
                    # api_type='OpenAI',
                    api_key=openai_api_key,
                    api_base=openai_compatible_url, #'http://llmhost1:8080/v1',
                    api_version=openai_api_version, #'v1',
                    model_name=model_name #'jina-embeddings-v4-text-retrieval-GGUF',
                )

def init_chroma_db(collection_name: str, 
                   collection_path: str,
                   embedding_function: EmbeddingFunction
                ) -> tuple[chromadb.PersistentClient, ChromaCollection]:
    # Locate and intiialize ChromaDB vector store if needed
    logger.debug("Initializing ChromaDB client for path %s", collection_path)
    chroma_client = chromadb.PersistentClient(path=collection_path)
    logger.debug("Looking for Chroma collection %s", collection_name)
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        logger.debug("Found existing Chroma collection at %s", collection_path)
    except Exception as e:
        logger.warning(f"Collection {collection_name} not found, creating new one: {e}")
        collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

    return chroma_client, collection

def compute_page_embeddings(page_data: dict, embedding_function: EmbeddingFunction) -> Embeddings:
    logger.debug("Computing embedding for page(%d) titled %s", page_data['page_id'], page_data['title'])
    text_content = f"{page_data['title']}\n{page_data['abstract']}"
    return embedding_function(text_content)

def test_one_embedding():
    sqlconn = get_sql_conn()
    embedding_function = get_embedding_function(
        model_name="jina-embeddings-v4-text-matching-GGUF",
        openai_compatible_url='http://llmhost1.internal.tajh.house:8080/v1',
        openai_api_key='no-key-necessary'
    )

    page_data = get_any_page(sqlconn)
#    logger.debug(f"Fetched page data: {json.dumps(page_data)}")

    embeddings = compute_page_embeddings(page_data, embedding_function)
    # logger.debug(f"Computed embeddings:\n{embeddings}")
    # logger.debug(f"Vector count: {len(embeddings)}")
    # n = 0
    # for vector in embeddings:
    #     n += 1
    #     logger.info(f"Vector {n} length: {len(vector)}.")

    # store the embedding
    if len(embeddings) > 1:
        logger.warning("More than one embedding returned, only storing the first one.")
    page_data['embedding_vector'] = embeddings[0]

    update_embeddings_for_page(page_data, sqlconn)

    # retrieve the embedding
    retrieved_page_data = get_page_by_id(page_data['page_id'], sqlconn)

    assert retrieved_page_data['embedding_vector'] is not None, "Embedding vector was not stored correctly."
    assert retrieved_page_data['embedding_vector'].shape == page_data['embedding_vector'].shape, "Stored embedding vector shape does not match."
    assert (retrieved_page_data['embedding_vector'] == page_data['embedding_vector']).all(), "Stored embedding vector data does not match."


@contextmanager
def timer():
    start_time = time.perf_counter()
    yield lambda: time.perf_counter() - start_time

def compute_embeddings_for_chunk(chunk_name: str, embedding_function: EmbeddingFunction, sqlconn: sqlite3.Connection) -> None:

    logger.info("Computing embeddings for chunk %s", chunk_name)
    with timer() as elapsed:
        page_id_list = get_page_ids_needing_embedding_for_chunk(chunk_name, sqlconn)
        logger.info("Found %d pages needing embeddings in chunk %s", len(page_id_list), chunk_name)
        counter = 0
        for page_id in page_id_list:
            page_data = get_page_by_id(page_id, sqlconn)
            embeddings = compute_page_embeddings(page_data, embedding_function)
            if len(embeddings) > 1:
                logger.warning("More than one embedding returned, only storing the first one.")
            page_data['embedding_vector'] = embeddings[0]
            update_embeddings_for_page(page_data, sqlconn)
            logger.debug("Stored embedding for page_id %d", page_id)
            counter += 1
    logger.info("Computed and stored embeddings for %d pages in chunk %s in %.2f seconds", counter, chunk_name, elapsed())

        
if __name__ == "__main__":
    sqlconn = get_sql_conn()
    jina_text_matching_function = get_embedding_function(
        model_name="jina-embeddings-v4-text-matching-GGUF",
        openai_compatible_url='http://llmhost1.internal.tajh.house:8080/v1',
        openai_api_key='no-key-necessary'
    )
    enwiki_ns_0 = "enwiki_namespace_0"
    enwiki_chunk_0 = "enwiki_namespace_0_chunk_0"
    compute_embeddings_for_chunk(chunk_name=enwiki_chunk_0, embedding_function=jina_text_matching_function, sqlconn=sqlconn)
