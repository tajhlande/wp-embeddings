import contextlib
import io
import json
import logging
import os
import sqlite3
import tarfile

from dotenv import load_dotenv

# from wme_sdk.api.snapshot import Snapshot
from common import ensure_tables, get_sql_conn, upsert_new_chunk_data, upsert_new_page_data
from wme_sdk.auth.auth_client import AuthClient
from wme_sdk.api.api_client import Client, Request, Filter

# Load environment variables from .env file 
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
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
        logger.exception(f"Login failed: {e}")
        return

    refresh_token = login_response['refresh_token']
    access_token = login_response['access_token']
    return auth_client, refresh_token, access_token

def get_enterprise_api_client(access_token) -> Client:
    logger.debug("Creating API Client with access token")
    api_client = Client()
    api_client.set_access_token(access_token)
    return api_client
  

@contextlib.contextmanager
def revoke_token_on_exit(auth_client: Client, refresh_token: str):
    try:
        yield
    finally:
        try:
            auth_client.revoke_token(refresh_token)
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")

def find_chunks(api_client: Client, sqlconn: sqlite3.Connection, request: Request, namespace: str): 
    try:
        # "enwiki_namespace_0"
        snapshot_json = api_client.get_snapshot(namespace, request)
    except Exception as e:
        logger.exception(f"Failed to get snapshot")
        return

    try:
        for chunk_name in snapshot_json['chunks']:
            logger.info(f"Chunk: {chunk_name}")
            upsert_new_chunk_data(chunk_name, namespace, sqlconn)
            # logger.info(f"Name: {chunk_json['date_modified']}")
            # logger.info(f"Abstract: {chunk_json['identifier']}")
            # logger.info(f"Description: {chunk_json['size']}")  
    except Exception as e:
        logger.exception(f"Failed to process snapshot data: {e}")
        logger.info(f"Snapshot JSON:\n{json.dumps(snapshot_json, indent=2)}")
        return

def download_chunk(api_client: Client, 
                   namespace: str, # "enwiki_namespace_0"
                   chunk_name: str, # "enwiki_namespace_0_chunk_0"
                   chunk_file_path: str, # "enwiki_namespace_0_chunk_0.tar.gz"
                   ):
    try:
        # Create a BytesIO object to hold the downloaded content
        chunk_writer = io.BytesIO()
        api_client.download_chunk(namespace, chunk_name, chunk_writer)
        # Save the content to a file
        chunk_data = chunk_writer.getvalue()
        logger.info(f"Downloaded {len(chunk_data)} bytes")
        with open(chunk_file_path, "wb") as f:
            f.write(chunk_data)
        logger.info("Chunk data saved to downloaded_chunk.tar.gz")

    except Exception as e:
        logger.exception(f"Failed to download chunk data: {e}")
        return



def extract_single_file_from_tar_gz(tar_gz_path: str, extract_to: str='.'):
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
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
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
        return None

# Usage
# extracted_file = extract_single_file_from_tar_gz('archive.tar.gz', './output/')
# if extracted_file:
#     print(f"Successfully extracted: {extracted_file}")

def parse_chunk_file(sqlconn: sqlite3.Connection, chunk_name: str, chunk_file_path: str) -> int:
    """
    Parse the extracted chunk file to read its contents.
    
    Args:
        chunk_file_path (str): Path to the extracted chunk file.
    
    """
    try:
        logger.info(f"Parsing chunk file: {chunk_file_path}")
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                # Assuming each line is a JSON object representing a page
                #logger.debug(f"Reading line {line_number}: {line.strip()[:1000]}")
                raw_page_data = json.loads(line)
                #logger.debug(f"Parsed JSON: {json.dumps(page_data)[:1000]}")
                page_data_extract = {
                    'page_id': raw_page_data.get('id'),
                    'title': raw_page_data.get('name'),
                    'chunk_name': chunk_name,
                    'url': raw_page_data.get('url'),
                    'abstract': raw_page_data.get('abstract'),
                }
                #logger.debug(f"Extracted page data: {json.dumps(page_data_extract, indent=2)}")
                logger.debug(f"Processing page on line {line_number}. Page ID: {page_data_extract['page_id']}, Page title: {page_data_extract['title']}")
                upsert_new_page_data(page_data_extract, sqlconn)
        return line_number
                  
                
    except Exception as e:
        logger.exception(f"Error parsing chunk file: {e}")
        raise
       
 

def fetch_and_extract():

    auth_client, refresh_token, access_token = get_enterprise_auth_client()

    with revoke_token_on_exit(auth_client, refresh_token):
        api_client = get_enterprise_api_client(access_token)

       #to get metadata of all available snapshots

        # try:
        #     snapshots = api_client.get_snapshots(Request())
        # except Exception as e:
        #     logger.exception(f"Failed to get snapshots: {e}")
        #     return

        # for content in snapshots:
        #     logger.info(f"Name: {content['date_modified']}")
        #     logger.info(f"Abstract: {content['identifier']}")
        #     logger.info(f"Description: {content['size']}")
        
        # To get metadata on an single SC snapshot using request parameters   
        request = Request(
            filters=[Filter(field="in_language.identifier", value="en")]
        )

        # try:
        #     ss_json = api_client.get_structured_snapshot("enwiki_namespace_0", request)
        # except Exception as e:
        #     logger.exception(f"Failed to get structured content")
        #     return


        # try:
        #     head = api_client.head_structured_snapshot("enwiki_namespace_0")
        # except Exception as e:
        #     logger.exception(f"Failed to head structured content")
        #     return
        # logger.info(f"Head of structured snapshot:\n{json.dumps(head, indent=2)}")


        try:
            chunk_info_json = api_client.get_chunk("enwiki_namespace_0", "enwiki_namespace_0_chunk_0", request)
        except Exception as e:
            logger.exception(f"Failed to get chunk info")
            return
        logger.info(f"Chunk info:\n{json.dumps(chunk_info_json, indent=2)}")


def process_one_chunk():
    # auth_client, refresh_token, access_token = get_enterprise_auth_client()

    # with revoke_token_on_exit(auth_client, refresh_token):
    #     api_client = get_enterprise_api_client(access_token)

        namespace = "enwiki_namespace_0"
        chunk_name = "enwiki_namespace_0_chunk_0"
        chunk_file_path = f"downloaded/{namespace}/{chunk_name}.tar.gz"

        #download_chunk(api_client, namespace, chunk_name, chunk_file_path)

        logger.info(f"Path for extracted chunk archive: {chunk_file_path}")
        extracted_chunk_path = f"extracted/{namespace}"

        extracted_file_name = extract_single_file_from_tar_gz(chunk_file_path, extract_to=extracted_chunk_path)
        if extracted_file_name:
            logger.info(f"Extracted chunk file: {extracted_file_name}")
        else:
            logger.error("Failed to extract file from chunk")

        sqlconn = get_sql_conn()
        parse_chunk_file(sqlconn, chunk_name, os.path.join(extracted_chunk_path, extracted_file_name))

def get_chunk_info_for_namespace(namespace: str, api_client: Client, sqlconn: sqlite3.Connection):
    logger.info(f"Fetching chunk metadata for namespace: {namespace}")
    request = Request(
        filters=[Filter(field="in_language.identifier", value="en")]
    )
    chunk_data_list: list[dict] = api_client.get_chunks(namespace, request)
    for chunk_data in chunk_data_list:
        #logger.info(f"Found chunk: {json.dumps(chunk_data)}")
        chunk_name = chunk_data.get('identifier')
        if chunk_name:
            upsert_new_chunk_data(chunk_name, namespace, sqlconn)
        else:
            logger.warning(f"Chunk data without a name field: {json.dumps(chunk_data)}")
    logger.info(f"Total chunks found: {len(chunk_data_list)}")

if __name__ == "__main__":
    sqlconn = get_sql_conn()
    ensure_tables(sqlconn)
    #process_one_chunk()
    auth_client, refresh_token, access_token = get_enterprise_auth_client()

    with revoke_token_on_exit(auth_client, refresh_token):
        api_client = get_enterprise_api_client(access_token)
        get_chunk_info_for_namespace("enwiki_namespace_0", api_client, sqlconn)