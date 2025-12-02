import contextlib
import io
import json
import logging
import sqlite3
import tarfile
from typing import Optional

from dotenv import load_dotenv

# from wme_sdk.api.snapshot import Snapshot
from classes import Chunk, Page
from database import (
    ensure_tables,
    get_sql_conn,
    upsert_new_chunk_data,
    upsert_new_pages_in_batch,
)
from wme_sdk.auth.auth_client import AuthClient
from wme_sdk.api.api_client import Client, Request, Filter
from progress_utils import ProgressTracker

# Load environment variables from .env file
load_dotenv()

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Apply import fix before importing wme_sdk_python modules
# try:
#     import fix_imports
#     logger.info("Import fix applied successfully")
# except Exception as e:
#     logger.warning(f"Could not apply import fix: {e}")

# # Ensure the current directory is in Python path for relative imports in wme_sdk_python
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# # Debug logging to diagnose import issues
# logger.info(f"Python path: {sys.path}")
# logger.info(f"Current working directory: {os.getcwd()}")
# logger.info(f"PYTHONPATH environment variable: {os.getenv('PYTHONPATH')}")

# Now try to import the modules
# try:

#     logger.info("Successfully imported all required modules")
# except Exception as e:
#     logger.error(f"Failed to import modules: {e}")
#     logger.error(f"Python path contents: {sys.path}")
#     raise


def get_enterprise_auth_client() -> tuple[AuthClient, str, str]:
    logger.debug("Creating AuthClient and logging in")
    # Load environment variables from .env file
    load_dotenv()
    auth_client = AuthClient()
    try:
        login_response = auth_client.login()
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise

    refresh_token = login_response["refresh_token"]
    access_token = login_response["access_token"]
    return auth_client, refresh_token, access_token


def get_enterprise_api_client(access_token) -> Client:
    logger.debug("Creating API Client with access token")
    api_client = Client(download_chunk_size=1024 * 1024, download_concurrency=4)
    api_client.set_access_token(access_token)
    return api_client


@contextlib.contextmanager
def revoke_token_on_exit(auth_client: AuthClient, refresh_token: str):
    try:
        yield
    finally:
        try:
            auth_client.revoke_token(refresh_token)
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")


def download_chunk(
    api_client: Client,
    namespace: str,  # "enwiki_namespace_0"
    chunk_name: str,  # "enwiki_namespace_0_chunk_0"
    chunk_file_path: str,  # "enwiki_namespace_0_chunk_0.tar.gz"
    tracker: Optional[ProgressTracker] = None,
):
    try:
        # Create a BytesIO object to hold the downloaded content
        chunk_writer = io.BytesIO()
        api_client.download_chunk(namespace, chunk_name, chunk_writer, tracker)
        # Save the content to a file
        chunk_data = chunk_writer.getvalue()
        # logger.info(f"Downloaded {len(chunk_data)} bytes")
        with open(chunk_file_path, "wb") as f:
            f.write(chunk_data)

    except Exception as e:
        logger.exception(f"Failed to download chunk data: {e}")
        return


def extract_single_file_from_tar_gz(tar_gz_path: str, extract_to: str = "."):
    """
    Extract a single file from a tar.gz archive

    Args:
        tar_gz_path (str): Path to the tar.gz file
        extract_to (str): Directory to extract to

    Returns:
        str: Name of the extracted file
    """
    try:
        logger.debug(f"Extracting from {tar_gz_path} to {extract_to}")
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            members = tar.getmembers()

            if not members:
                raise ValueError("tar.gz archive is empty")

            # Get first member (assuming it's the only file)
            first_member = members[0]

            if first_member.isfile():
                # Extract the file
                tar.extract(first_member, path=extract_to)
                return first_member.name
            else:
                raise ValueError("First member is not a regular file")

    except Exception as e:
        print(f"Error extracting archive: {e}")
        raise


# Usage
# extracted_file = extract_single_file_from_tar_gz('archive.tar.gz', './output/')
# if extracted_file:
#     print(f"Successfully extracted: {extracted_file}")


def count_lines_in_file(file_path: str) -> int:
    """
    Count the number of lines in a given file.

    Args:
        file_path (str): Path to the file to count lines in.

    Returns:
        int: Number of lines in the file.
    """
    try:
        line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
        return line_count
    except Exception as e:
        logger.exception(f"Error counting lines in file {file_path}: {e}")
        raise


def parse_chunk_file(
    sqlconn: sqlite3.Connection,
    namespace: str,
    chunk_name: str,
    chunk_file_path: str,
    tracker: Optional[ProgressTracker] = None,
) -> int:
    """
    Parse the extracted chunk file to read its contents.

    Args:
        chunk_file_path (str): Path to the extracted chunk file.

    """
    try:
        # logger.info(f"Parsing chunk file: {chunk_file_path}")

        # First, count total lines for progress tracking
        count_lines_in_file(chunk_file_path)
        # logger.info(f"Found {total_lines} lines to process")

        # Parse with progress bar - reduce logging frequency to avoid interference

        BUFFER_SIZE = 1000
        buffer = []
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            line_number = 0
            for line in f:
                line_number += 1
                # Assuming each line is a JSON object representing a page
                raw_page_data = json.loads(line)
                page_id = raw_page_data.get("identifier")
                if page_id is None:
                    page_id = raw_page_data.get("id")
                if page_id is None:
                    raise ValueError(f"Can't get page id from raw page: {line[:100]}...")
                page = Page(
                    namespace=namespace,
                    page_id=page_id,
                    title=raw_page_data.get("name"),
                    chunk_name=chunk_name,
                    url=raw_page_data.get("url"),
                    abstract=raw_page_data.get("abstract"),
                )
                buffer.append(page)
                if len(buffer) >= BUFFER_SIZE:
                    upsert_new_pages_in_batch(buffer, sqlconn, BUFFER_SIZE)
                    tracker.update(len(buffer)) if tracker else None
                    buffer = []
        # flush remainder
        if buffer:
            upsert_new_pages_in_batch(buffer, sqlconn, BUFFER_SIZE)
            tracker.update(len(buffer)) if tracker else None

        # with open(chunk_file_path, "r", encoding="utf-8") as f:
        #     line_number = 0
        #     for line in f:
        #         line_number += 1
        #         # Assuming each line is a JSON object representing a page
        #         raw_page_data = json.loads(line)
        #         page_id = raw_page_data.get("identifier")
        #         if page_id is None:
        #             page_id = raw_page_data.get("id")
        #         if page_id is None:
        #             raise ValueError(f"Can't get page id from raw page: {line[:100]}...")
        #         page = Page(
        #             page_id=page_id,
        #             title=raw_page_data.get("name"),
        #             chunk_name=chunk_name,
        #             url=raw_page_data.get("url"),
        #             abstract=raw_page_data.get("abstract"),
        #         )

        #         # Log progress if no tracker
        #         if not tracker and line_number % 10000 == 0:
        #             logger.info(f"Processed {line_number} lines so far...")

        #         # Upsert page data into the database
        #         upsert_new_page_data(page, sqlconn)
        #         tracker.update(1) if tracker else None

        # logger.info(f"Completed parsing {chunk_name}: {line_number} lines processed")
        return line_number

    except Exception as e:
        logger.exception(f"Error parsing chunk file: {e}")
        raise


def get_chunk_info_for_namespace(
    namespace: str, api_client: Client, sqlconn: sqlite3.Connection
):
    logger.info(f"Fetching chunk metadata for namespace: {namespace}")
    request = Request(filters=[Filter(field="in_language.identifier", value="en")])
    chunk_data_list: list[dict] = api_client.get_chunks(namespace, request)
    chunk_list = [
        Chunk(chunk_name=chunk_data.get("identifier"), namespace=namespace)  # type: ignore
        for chunk_data in chunk_data_list
        if chunk_data.get("identifier")
    ]
    for chunk in chunk_list:
        # logger.info(f"Found chunk: {json.dumps(chunk_data)}")
        upsert_new_chunk_data(chunk, sqlconn)
    logger.info(f"Total chunks found: {len(chunk_data_list)}")


if __name__ == "__main__":
    sqlconn = get_sql_conn("enwiki_namespace_0")
    ensure_tables(sqlconn)
    # process_one_chunk()
    logger.info("Getting enterprise auth client")
    auth_client, refresh_token, access_token = get_enterprise_auth_client()

    with revoke_token_on_exit(auth_client, refresh_token):
        logger.info("Getting enterprise api client")
        api_client = get_enterprise_api_client(access_token)
        # et_chunk_info_for_namespace("enwiki_namespace_0", api_client, sqlconn)

        # get list of namespaces
        request = Request()  # filters=[Filter(field="in_language.identifier", value="en")])
        namespaces = api_client.get_namespaces(request)
        print(f"Found {len(namespaces)} namespaces")
        for d in namespaces:
            print(f"{json.dumps(d)}")
