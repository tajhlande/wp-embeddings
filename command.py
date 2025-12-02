import argparse
import atexit
import json
import logging
import os
import readline

import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from itertools import batched
from typing import Dict, List, Any

# Import existing modules
from database import (
    delete_cluster_tree,
    get_all_cluster_nodes_for_topic_labeling,
    get_embedding_count,
    get_pages_in_all_clusters,
    get_sql_conn,
    ensure_tables,
    get_page_ids_needing_embedding_for_chunk,
    remove_existing_cluster_topics,
    update_cluster_tree_final_labels,
    update_cluster_tree_first_labels,
)
from download_chunks import (
    count_lines_in_file,
    get_enterprise_auth_client,
    get_enterprise_api_client,
    get_chunk_info_for_namespace,
    download_chunk,
    extract_single_file_from_tar_gz,
    parse_chunk_file,
)
from index_pages import (
    get_embedding_function,
    compute_embeddings_for_chunk,
    get_embedding_model_config,
)
from languages import get_language_for_namespace
from progress_utils import ProgressTracker
from topic_discovery import TopicDiscovery
from transform import run_pca, run_umap_per_cluster, run_recursive_clustering

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Store the command history (feel free to change the path)
_HISTFILE = os.path.join(os.path.expanduser("."), ".wp_transform_history")

# Load existing history, ignore errors if the file does not exist yet
if os.path.exists(_HISTFILE):
    readline.read_history_file(_HISTFILE)

# Save the history when the interpreter exits
atexit.register(readline.write_history_file, _HISTFILE)

# Limit the size of the history file
readline.set_history_length(1000)  # keep last 1000 lines


class Argument():
    """Description of a command line argument"""
    def __init__(
        self,
        name: str,
        description: str,
        type: str,
        required: bool,
        default: Any = None
    ):
        self.name = name
        self.description = description
        self.type = type
        self.required = required
        self.default = default


class Result(IntEnum):
    SUCCESS = 0
    FAILURE = -1


REQUIRED_NAMESPACE_ARGUMENT = Argument(name="namespace", type="string", required=True,
                                       description="The wiki namespace, e.g. enwiki_namespace_0")
OPTIONAL_CHUNK_LIMIT_ARGUMENT = Argument(name="limit", type="integer", required=False, default=1,
                                         description="Number of chunks to process")
OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT = Argument(name="limit", type="integer", required=False,
                                                    description="Number of chunks to process")
OPTIONAL_CHUNK_NAME_ARGUMENT = Argument(name="chunk", type="string", required=False,
                                        description="The name of the chunk to process")
OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT = Argument(
    name="limit", type="integer", required=False, description="Number of pages to process"
)
OPTIONAL_BATCH_SIZE_ARGUMENT = Argument(
    name="batch", type="integer", required=False, default=100, description="Size of batch to process at a time"
)
REQUIRED_MODE_ARGUMENT = Argument(
    name="mode", type="string", required=True, description="Processing mode: 'refresh' or 'resume'"
)
OPTIONAL_LIMIT_ARGUMENT = Argument(
    name="limit", type="integer", required=False, description="Maximum number of clusters to process across all passes"
)

CHECK = "✓"
X = "✗"


class Command(ABC):
    """Abstract base class for all commands."""

    def __init__(
        self,
        name: str,
        description: str,
        expected_args: List[Argument] = []
    ):
        self.name = name
        self.description = description
        self.expected_args = expected_args

    def get_required_args(self):
        return [arg for arg in self.expected_args if arg.required is True]

    def get_optional_args(self):
        return [arg for arg in self.expected_args if arg.required is False]

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        """Execute the command with given arguments."""
        pass

    def validate(self, args: Dict[str, Any]) -> bool:
        """Validate command arguments."""
        logger.debug("Arguments: %s", json.dumps(args, indent=2))
        # Check required arguments
        for arg in self.expected_args:
            if arg.required and (arg.name not in args or args[arg.name] is None):
                raise ValueError(f"Missing required argument: {arg.name}")

        # Check argument types and values
        for expected_arg in self.expected_args:
            if expected_arg.name in args and args[expected_arg.name] is not None:
                supplied_arg = args[expected_arg.name]
                # Add specific validation rules here
                if expected_arg.type == "integer":
                    try:
                        val = int(supplied_arg)
                    except ValueError:
                        raise ValueError(f"Argument {expected_arg.name} must be an integer")
                    # Only apply >= 1 validation for arguments that should be positive integers
                    if expected_arg.name in [
                        CLUSTER_COUNT_ARGUMENT.name,
                        OPTIONAL_BATCH_SIZE_ARGUMENT.name,
                        LEAF_TARGET_ARGUMENT.name,
                        MAX_K_ARGUMENT.name,
                        MAX_DEPTH_ARGUMENT.name,
                        OPTIONAL_LIMIT_ARGUMENT.name
                    ]:
                        if val < 1:
                            raise ValueError(f"Argument {expected_arg.name} must be >= 1")
                elif expected_arg.type == "float":
                    try:
                        val = float(supplied_arg)
                    except ValueError:
                        raise ValueError(f"Argument {expected_arg.name} must be a float")
                    # Only apply >= 0 validation for arguments that should be non-negative
                    if expected_arg.name == "min-silhouette":
                        if val < 0:
                            raise ValueError(f"Argument {expected_arg.name} must be >= 0")
                elif expected_arg.type == "string":
                    if expected_arg.name == "mode":
                        if supplied_arg not in ["refresh", "resume"]:
                            raise ValueError(f"Argument {expected_arg.name} must be either 'refresh' or 'resume'")
            elif expected_arg.name not in args and expected_arg.default is not None:
                args[expected_arg.name] = expected_arg.default
        return True

    def format_argument_info(self, arg: Argument):
        return f"  --{arg.name} ({arg.type}{f', default: {arg.default}' if arg.default else ''}) {arg.description}\n"

    def get_help(self) -> str:
        """Get help text for the command."""
        help_text = f"Command \"{self.name}\": {self.description}\n"
        required_arguments = [arg for arg in self.expected_args if arg.required]
        optional_arguments = [arg for arg in self.expected_args if not arg.required]
        help_text += f"\nRequired arguments: {'' if required_arguments else 'None'}\n"
        for arg in required_arguments:
            help_text += self.format_argument_info(arg)
        help_text += f"\nOptional arguments: {'' if optional_arguments else 'None'}\n"
        for arg in optional_arguments:
            help_text += self.format_argument_info(arg)
        return help_text


class CommandParser:
    """Parse user input into command and arguments."""

    def __init__(self):
        self.commands: dict[str, Command] = {}

    def register_command(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command

    def parse_input(self, input_str: str) -> tuple[str, Dict[str, Any]]:
        """Parse user input into command and arguments."""
        input_str = input_str.strip()

        if not input_str:
            return "", {}

        # Split into command and arguments
        parts = input_str.split()
        if not parts:
            return "", {}

        command_name = parts[0].lower()
        args = {}

        # Parse arguments
        for i in range(1, len(parts)):
            part = parts[i]
            if part.startswith("--"):
                # Long argument --value
                if "=" in part:
                    key, value = part[2:].split("=", 1)
                    args[key] = self._convert_value(value)
                else:
                    key = part[2:]
                    # Check if next part is a value
                    if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                        args[key] = self._convert_value(parts[i + 1])
                        i += 1  # Skip next part as it's been consumed
                    else:
                        args[key] = True
            elif part.startswith("-"):
                # Short argument -v
                key = part[1:]
                if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    args[key] = self._convert_value(parts[i + 1])
                    i += 1  # Skip next part as it's been consumed
                else:
                    args[key] = True
            else:
                # Positional argument - treat as value for 'command' argument for help command
                # or as a positional argument for other commands
                if command_name == "help":
                    args["command"] = part
                else:
                    # For other commands, we could handle positional arguments here
                    # For now, just ignore or handle as needed
                    pass

        return command_name, args

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def validate_command(self, command_name: str) -> bool:
        """Check if command exists."""
        return command_name in self.commands

    def get_command_help(self, command_name: str) -> tuple[Result, str]:
        """Get help for a specific command."""
        if command_name in self.commands:
            return Result.SUCCESS, self.commands[command_name].get_help()
        return Result.FAILURE, f"Unknown command: {command_name}"

    def get_available_commands(self) -> List[str]:
        """Get list of available commands."""
        return list(self.commands.keys())


class CommandDispatcher:
    """Dispatch commands to appropriate command classes."""

    def __init__(self, parser: CommandParser):
        self.parser = parser
        self.api_client = None
        self.auth_client = None
        self.refresh_token = None
        self.access_token = None

    def _get_api_client(self):
        """Get or create API client with authentication."""
        if self.api_client is None:
            try:
                self.auth_client, self.refresh_token, self.access_token = (
                    get_enterprise_auth_client()
                )
                self.api_client = get_enterprise_api_client(self.access_token)
            except Exception as e:
                logger.error(f"Failed to authenticate: {e}")
                raise
        return self.api_client

    def dispatch(self, command_name: str, args: Dict[str, Any]) -> tuple[Result, str]:
        """Dispatch command to appropriate handler."""
        if not self.parser.validate_command(command_name):
            return Result.FAILURE, f"Unknown command: {command_name}"

        try:
            command = self.parser.commands[command_name]
            command.validate(args)
            return command.execute(args)
        except ValueError as e:
            return Result.FAILURE, f"Validation error: {e}"
        except Exception as e:
            logger.exception(f"Error executing command '{command_name}': {e}")
            return Result.FAILURE, f"Error: {e}"


class RefreshChunkDataCommand(Command):
    """Refresh chunk data for a namespace."""

    def __init__(self):
        super().__init__(
            name="refresh",
            description="Refresh chunk data for a namespace",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
        logger.info("Refreshing chunk data for namespace: %s", namespace)

        try:
            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            api_client = self._get_api_client()
            get_chunk_info_for_namespace(namespace, api_client, sqlconn)

            # Get count of chunks for the namespace
            cursor = sqlconn.execute(
                "SELECT COUNT(*) as count FROM chunk_log WHERE namespace = ?",
                (namespace,),
            )
            count = cursor.fetchone()["count"]

            return (
                Result.SUCCESS,
                f"{CHECK} Refreshed chunk data for namespace: {namespace}\n"
                f"Found {count} chunks in namespace {namespace}"
            )

        except Exception as e:
            logger.error(f"Failed to refresh chunk data: {e}")
            return Result.FAILURE, f"{X} Failed to refresh chunk data: {e}"

    def _get_api_client(self):
        """Get or create API client with authentication."""
        if not hasattr(self, "api_client") or self.api_client is None:
            try:
                auth_client, refresh_token, access_token = get_enterprise_auth_client()
                self.api_client = get_enterprise_api_client(access_token)
            except Exception as e:
                logger.error(f"Failed to authenticate: {e}")
                raise
        return self.api_client


class DownloadChunksCommand(Command):
    """Download chunks that haven't been downloaded yet."""

    def __init__(self):
        super().__init__(
            name="download",
            description="Download chunks that haven't been downloaded yet",
            expected_args=[OPTIONAL_CHUNK_LIMIT_ARGUMENT, REQUIRED_NAMESPACE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        limit = args.get(OPTIONAL_CHUNK_LIMIT_ARGUMENT.name, OPTIONAL_CHUNK_LIMIT_ARGUMENT.default)
        namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]

        try:
            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            # Get chunks that need downloading
            if namespace:
                cursor = sqlconn.execute(
                    """
                    SELECT chunk_name, namespace
                    FROM chunk_log
                    WHERE namespace = ?
                    AND downloaded_at IS NULL
                    ORDER BY chunk_name ASC LIMIT ?;
                    """,
                    (namespace, limit),
                )
            else:
                cursor = sqlconn.execute(
                    """
                    SELECT chunk_name, namespace
                    FROM chunk_log
                    WHERE downloaded_at IS NULL
                    ORDER BY namespace ASC, chunk_name ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

            chunks_to_download = cursor.fetchall()

            if not chunks_to_download:
                return Result.FAILURE, f"{X} No chunks available for download"

            api_client = self._get_api_client()
            downloaded_count = 0

            if limit:
                logger.info("Limiting download to %d chunks at most", limit)
                chunks_to_download = chunks_to_download[:limit]
                logger.info(
                    "Planning to download %d chunks (limited)", len(chunks_to_download)
                )
            else:
                logger.info("Planning to download %d chunks", len(chunks_to_download))

            for chunk in chunks_to_download:
                chunk_name = chunk["chunk_name"]
                chunk_namespace = chunk["namespace"]
                # logger.info("Downloading chunk %s from namespace %s...", chunk_name, chunk_namespace)

                try:
                    # Create download directory
                    download_dir = f"downloaded/{chunk_namespace}"
                    os.makedirs(download_dir, exist_ok=True)

                    chunk_file_path = f"{download_dir}/{chunk_name}.tar.gz"

                    # Download chunk
                    with ProgressTracker(
                        f"Downloading {chunk_name}", unit="bytes"
                    ) as tracker:
                        download_chunk(
                            api_client,
                            chunk_namespace,
                            chunk_name,
                            chunk_file_path,
                            tracker,
                        )
                    logger.debug("Chunk saved to %s", chunk_file_path)

                    # Update database
                    sqlconn.execute(
                        """
                        UPDATE chunk_log
                        SET chunk_archive_path = ?, downloaded_at = CURRENT_TIMESTAMP
                        WHERE chunk_name = ?
                        """,
                        (chunk_file_path, chunk_name),
                    )
                    sqlconn.commit()

                    downloaded_count += 1
                    logger.debug("Downloaded %s", chunk_name)

                except Exception as e:
                    logger.error(f"Failed to download {chunk_name}: {e}")
                    continue

            return Result.SUCCESS, f"{CHECK} Downloaded {downloaded_count} chunk(s) in namespace {namespace or 'all'}"

        except Exception as e:
            logger.error(f"Failed to download chunks: {e}")
            return Result.FAILURE, f"{X} Failed to download chunks: {e}"

    def _get_api_client(self):
        """Get or create API client with authentication."""
        if not hasattr(self, "api_client") or self.api_client is None:
            try:
                auth_client, refresh_token, access_token = get_enterprise_auth_client()
                self.api_client = get_enterprise_api_client(access_token)
            except Exception as e:
                logger.error(f"Failed to authenticate: {e}")
                raise
        return self.api_client


class UnpackProcessChunksCommand(Command):
    """Unpack and process downloaded chunks."""

    def __init__(self):
        super().__init__(
            name="unpack",
            description="Unpack and process downloaded chunks",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT, OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
        limit = args.get(OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT.name)

        try:
            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            # Get chunks that need unpacking
            if namespace:
                cursor = sqlconn.execute(
                    """
                    SELECT chunk_name, namespace, chunk_archive_path
                    FROM chunk_log
                    WHERE namespace = ?
                    AND chunk_archive_path IS NOT NULL
                    AND chunk_extracted_path IS NULL
                    ORDER BY chunk_name ASC
                    """,
                    (namespace,),
                )
            else:
                cursor = sqlconn.execute(
                    """
                    SELECT chunk_name, namespace, chunk_archive_path
                    FROM chunk_log
                    WHERE chunk_archive_path IS NOT NULL
                    AND chunk_extracted_path IS NULL
                    ORDER BY namespace ASC, chunk_name ASC
                    """
                )

            chunks_to_unpack = cursor.fetchall()

            if not chunks_to_unpack:
                return Result.FAILURE, f"{X} No chunks available for unpacking"

            processed_count = 0
            total_pages = 0

            if limit:
                chunks_to_unpack = chunks_to_unpack[:limit]
                logger.info("Limiting unpacking to %d chunks at most", limit)

            logger.info("Planning to unpack to %d chunks", len(chunks_to_unpack))

            for chunk in chunks_to_unpack:
                chunk_name = chunk["chunk_name"]
                chunk_namespace = chunk["namespace"]
                archive_path = chunk["chunk_archive_path"]
                logger.debug(
                    "Unpacking chunk %s from namespace %s...",
                    chunk_name,
                    chunk_namespace,
                )

                try:
                    # Create extraction directory
                    extract_dir = f"extracted/{chunk_namespace}"
                    os.makedirs(extract_dir, exist_ok=True)

                    # Extract archive
                    logger.info(
                        "Extracting archive %s to directory %s...",
                        archive_path,
                        extract_dir,
                    )
                    extracted_file = extract_single_file_from_tar_gz(
                        archive_path, extract_dir
                    )

                    if extracted_file:
                        chunk_file_path = f"{extract_dir}/{extracted_file}"

                        # count lines in chunk file
                        file_line_count = count_lines_in_file(chunk_file_path)

                        # Parse chunk file
                        # logger.info("Parsing pages out of chunk file %s", chunk_file_path)
                        with ProgressTracker(
                            "Parsing chunk file", total=file_line_count, unit="pages"
                        ) as tracker:
                            line_count = parse_chunk_file(
                                sqlconn, chunk_namespace, chunk_name, chunk_file_path, tracker
                            )
                        logger.debug(
                            "Parsed %d pages from chunk file %s",
                            line_count,
                            chunk_file_path,
                        )

                        # Update database
                        sqlconn.execute(
                            "UPDATE chunk_log SET chunk_extracted_path = ?, "
                            "unpacked_at = CURRENT_TIMESTAMP WHERE chunk_name = ?",
                            (chunk_file_path, chunk_name),
                        )
                        sqlconn.commit()

                        # Remove extracted file
                        logger.debug("Cleaning up extracted file %s", chunk_file_path)
                        try:
                            os.remove(chunk_file_path)
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove extracted file {chunk_file_path}: {e}"
                            )

                        # Count pages processed
                        page_cursor = sqlconn.execute(
                            "SELECT COUNT(*) as count FROM page_log WHERE chunk_name = ?",
                            (chunk_name,),
                        )
                        pages_in_chunk = page_cursor.fetchone()["count"]
                        total_pages += pages_in_chunk

                        processed_count += 1
                        logger.debug(
                            f"Unpacked and processed {chunk_name} with {pages_in_chunk} pages."
                        )
                    else:
                        logger.error(f"Failed to extract {archive_path}")

                except Exception as e:
                    logger.error(f"Failed to unpack {chunk_name}: {e}")
                    continue

            return (
                Result.SUCCESS,
                f"{CHECK} Unpacked and processed {processed_count} chunk(s). "
                f"Processed {total_pages} pages from {processed_count} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to unpack chunks: {e}")
            return Result.FAILURE, f"{X} Failed to unpack chunks: {e}"


class EmbedPagesCommand(Command):
    """Process remaining pages for embedding computation."""

    def __init__(self):
        super().__init__(
            name="embed",
            description="Process remaining pages for embedding computation",
            expected_args=[
                REQUIRED_NAMESPACE_ARGUMENT,
                OPTIONAL_CHUNK_NAME_ARGUMENT,
                OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT
            ]
        )
        embedding_model_name, embedding_model_api_url, embedding_model_api_key = (
            get_embedding_model_config()
        )
        self.embedding_model_name = embedding_model_name
        self.embedding_model_api_url = embedding_model_api_url
        self.embedding_model_api_key = embedding_model_api_key

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
        chunk_name = args.get(OPTIONAL_CHUNK_NAME_ARGUMENT.name)

        if chunk_name and not namespace:
            return Result.FAILURE, f"{X} --namespace is required when --chunk_name is provided"

        limit = args.get(OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT.name, OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT.default)

        logger.info(
            "Computing embeddings for pages%s%s%s...",
            f" in namespace {namespace}" if namespace else "",
            f" in chunk {chunk_name}" if chunk_name else "",
            f" with limit {limit}" if limit else "",
        )

        try:
            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            # Setup embedding function
            embedding_function = get_embedding_function(
                model_name=self.embedding_model_name,
                openai_compatible_url=self.embedding_model_api_url,
                openai_api_key=self.embedding_model_api_key,
            )

            if chunk_name:
                # Process specific chunk
                page_ids = get_page_ids_needing_embedding_for_chunk(chunk_name, sqlconn, namespace=namespace)

                if limit:
                    pages_to_process = min(len(page_ids), limit)
                    if limit < len(page_ids):
                        logger.debug(
                            "Pages to process: %d for chunk %s (limit applied)",
                            pages_to_process,
                            chunk_name,
                        )
                    else:
                        logger.debug(
                            "Pages to process: %d for chunk %s (limit not restricting)",
                            pages_to_process,
                            chunk_name,
                        )
                else:
                    pages_to_process = len(page_ids)
                    logger.debug(
                        "Pages to process: %d for chunk %s (not limited)",
                        pages_to_process,
                        chunk_name,
                    )

                if not page_ids:
                    return Result.FAILURE, f"{X} No pages needing embeddings in chunk {chunk_name}"

                # logger.info("Processing %d embeddings for named chunk %s...", len(page_ids), chunk_name)

                # Compute embeddings
                with ProgressTracker(
                    f"Computing embeddings for chunk {chunk_name}",
                    total=pages_to_process,
                    unit="pages",
                ) as tracker:
                    assert namespace is not None
                    compute_embeddings_for_chunk(
                        namespace=namespace,
                        chunk_name=chunk_name,
                        embedding_function=embedding_function,
                        sqlconn=sqlconn,
                        limit=pages_to_process,
                        tracker=tracker,
                    )

                return Result.SUCCESS, f"{CHECK} Processed {len(page_ids)} pages for chunk {chunk_name}"

            else:
                # Process all chunks
                logger.debug(
                    "Processing %sembeddings for next available chunk...",
                    f"up to {limit} " if limit else "",
                )

                if namespace:
                    sql = """
                        SELECT DISTINCT pl.namespace, pl.chunk_name
                        FROM page_log pl
                        LEFT JOIN page_vector pv ON pl.page_id = pv.page_id
                        WHERE pl.namespace = ? AND pv.embedding_vector IS NULL
                        ORDER BY pl.chunk_name ASC;
                    """
                    cursor = sqlconn.execute(sql, (namespace))
                else:
                    sql = """
                        SELECT DISTINCT pl.namespace, pl.chunk_name
                        FROM page_log pl
                        LEFT JOIN page_vector pv ON pl.page_id = pv.page_id
                        WHERE pv.embedding_vector IS NULL
                        ORDER BY pl.namespace ASC, pl.chunk_name ASC;
                    """
                    cursor = sqlconn.execute(sql)

                chunk_name_list = [(row["namespace"], row["chunk_name"]) for row in cursor.fetchall()]

                if not chunk_name_list:
                    return Result.FAILURE, f"{X} No pages needing embeddings"

                total_processed = 0

                for chunk_namespace, chunk_name in chunk_name_list:
                    try:
                        # Check if we have remaining limit capacity
                        if limit and total_processed >= limit:
                            break

                        # Get pages needing embeddings for this chunk
                        page_ids = get_page_ids_needing_embedding_for_chunk(
                            chunk_name=chunk_name, sqlconn=sqlconn, namespace=chunk_namespace,
                        )

                        if not page_ids:
                            continue

                        # Calculate how many pages we can process in this chunk
                        if limit:
                            remaining = limit - total_processed
                            pages_to_process = min(len(page_ids), remaining)
                            if pages_to_process < len(page_ids):
                                logger.debug(
                                    "Pages to process: %d for chunk %s (limit applied)",
                                    pages_to_process,
                                    chunk_name,
                                )
                            else:
                                logger.debug(
                                    "Pages to process: %d for chunk %s (limit not restricting)",
                                    pages_to_process,
                                    chunk_name,
                                )
                        else:
                            pages_to_process = len(page_ids)
                            logger.debug(
                                "Pages to process: %d for chunk %s (not limited)",
                                pages_to_process,
                                chunk_name,
                            )

                        if pages_to_process > 0:
                            # Pass the limit to the compute function
                            with ProgressTracker(
                                f"Computing embeddings for chunk {chunk_name}",
                                total=pages_to_process,
                                unit="pages",
                            ) as tracker:
                                compute_embeddings_for_chunk(
                                    namespace=chunk_namespace,
                                    chunk_name=chunk_name,
                                    embedding_function=embedding_function,
                                    sqlconn=sqlconn,
                                    limit=pages_to_process,
                                    tracker=tracker,
                                )
                            total_processed += pages_to_process
                            logger.debug(
                                f"Processed {pages_to_process} pages for chunk {chunk_name}"
                            )

                    except Exception as e:
                        logger.exception(f"Failed to process chunk {chunk_name}: {e}")
                        continue

                return (
                    Result.SUCCESS,
                    f"{CHECK} Embedded {total_processed} pages across {len(chunk_name_list)} chunk(s)"
                )

        except Exception as e:
            logger.exception(f"Failed to process pages: {e}")
            return Result.FAILURE, f"{X} Failed to process pages: {e}"


TARGET_DIMENSIONS_ARGUMENT = Argument(name="target-dim", type="integer", required=False, default=100,
                                      description="The target number of dimensions to reduce to")
BATCH_SIZE_ARGUMENT = Argument(name="batch-size", type="integer", required=False, default=10_000,
                               description="Number of pages to reduce per batch")


class ReduceCommand(Command):
    """Transform embedding vectors to a smaller vector space."""

    def __init__(self):
        super().__init__(
            name="reduce",
            description="Reduce dimension of embeddings",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT, TARGET_DIMENSIONS_ARGUMENT, BATCH_SIZE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]

            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            target_dim = args.get(TARGET_DIMENSIONS_ARGUMENT.name, TARGET_DIMENSIONS_ARGUMENT.default)
            batch_size = args.get(BATCH_SIZE_ARGUMENT.name, BATCH_SIZE_ARGUMENT.default)
            print(f"Target dimension: {target_dim}, batch size: {batch_size}")

            if target_dim > batch_size:
                return Result.FAILURE, f"{X} Batch size must be equal to or greater than target dimension."

            estimated_vector_count = get_embedding_count(namespace, sqlconn)
            estimated_batch_count = estimated_vector_count // batch_size + 1

            if estimated_vector_count < target_dim:
                return (
                    Result.FAILURE,
                    f"{X} Only found {estimated_vector_count} records, "
                    f"need at least {target_dim} to do PCA at that dimension"
                )

            if estimated_vector_count < batch_size:
                print(
                    "Reducing batch size to match estimated vector count: %d",
                    estimated_vector_count,
                )
                batch_size = estimated_vector_count

            with ProgressTracker(
                "PCA Reduction (two pass)",
                unit="batch",
                total=estimated_batch_count * 2,
            ) as tracker:
                batch_count, total_vector_count = run_pca(
                    sqlconn,
                    namespace=namespace,
                    target_dim=target_dim,
                    batch_size=batch_size,
                    tracker=tracker,
                )
            return (
                Result.SUCCESS,
                f"{CHECK} Reduced {total_vector_count} page embeddings in {batch_count} "
                f"batch{'' if batch_count == 1 else 'es'}"
            )
        except Exception as e:
            logger.exception(f"Failed to reduce embeddings: {e}")
            return Result.FAILURE, f"{X} Failed to reduce embeddings: {e}"


LEAF_TARGET_ARGUMENT = Argument(name="leaf-target", type="integer", required=False, default=50,
                                description="Target number of documents per leaf cluster")
MAX_K_ARGUMENT = Argument(name="max-k", type="integer", required=False, default=50,
                          description="Maximum number of clusters to create at each level")
MAX_DEPTH_ARGUMENT = Argument(name="max-depth", type="integer", required=False, default=10,
                              description="Maximum depth for recursion")
MIN_SILHOUETTE_ARGUMENT = Argument(name="min-silhouette", type="float", required=False, default=0.03,
                                   description="Minimum silhouette score to continue clustering")
CLUSTER_COUNT_ARGUMENT = Argument(name="clusters", type="integer", required=False, default=10_000,
                                  description="Number of clusters to process")
OPTIONAL_CLUSTER_LIMIT_ARGUMENT = Argument(name="limit", type="integer", required=False,
                                           description="Number of clusters to process")


class RecursiveClusterCommand(Command):
    """Run recursive clustering algorithm to build a tree of clusters."""

    def __init__(self):
        super().__init__(
            name="recursive-cluster",
            description="Run recursive clustering algorithm to build a tree of clusters",
            expected_args=[
                REQUIRED_NAMESPACE_ARGUMENT,
                LEAF_TARGET_ARGUMENT,
                MAX_K_ARGUMENT,
                MAX_DEPTH_ARGUMENT,
                MIN_SILHOUETTE_ARGUMENT,
                BATCH_SIZE_ARGUMENT
            ]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
            leaf_target = args.get(LEAF_TARGET_ARGUMENT.name, LEAF_TARGET_ARGUMENT.default)
            max_k = args.get(MAX_K_ARGUMENT.name, MAX_K_ARGUMENT.default)
            max_depth = args.get(MAX_DEPTH_ARGUMENT.name, MAX_DEPTH_ARGUMENT.default)
            min_silhouette = args.get(MIN_SILHOUETTE_ARGUMENT.name, MIN_SILHOUETTE_ARGUMENT.default)
            batch_size = args.get(BATCH_SIZE_ARGUMENT.name, BATCH_SIZE_ARGUMENT.default)

            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            print(f"Running recursive clustering on namespace {namespace}")
            print(f"Parameters: leaf-target={leaf_target}, max-k={max_k}, max-depth={max_depth}, "
                  f"min-silhouette={min_silhouette}, batch-size={batch_size}")

            delete_cluster_tree(sqlconn, namespace)

            with ProgressTracker(description="Recursive clustering", unit=" nodes") as tracker:
                nodes_created = run_recursive_clustering(
                    sqlconn,
                    namespace=namespace,
                    leaf_target=leaf_target,
                    max_k=max_k,
                    max_depth=max_depth,
                    min_silhouette_threshold=min_silhouette,
                    batch_size=batch_size,
                    tracker=tracker
                )

            return (
                Result.SUCCESS,
                f"{CHECK} Recursive clustering completed. Created {nodes_created} nodes "
                f"in cluster tree for namespace {namespace}."
            )
        except Exception as e:
            logger.exception(f"Failed to run recursive clustering: {e}")
            return Result.FAILURE, f"{X} Failed to run recursive clustering: {e}"


class TopicsCommand(Command):
    """Use an LLM to discover topics for clusters according to their page content."""

    def __init__(self):
        super().__init__(
            name="topics",
            description="Use an LLM to discover topics for clusters according to their page content.",
            expected_args=[
                REQUIRED_NAMESPACE_ARGUMENT,
                REQUIRED_MODE_ARGUMENT,
                OPTIONAL_LIMIT_ARGUMENT,
                OPTIONAL_BATCH_SIZE_ARGUMENT,
            ]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
            batch_size = args.get(OPTIONAL_BATCH_SIZE_ARGUMENT.name, OPTIONAL_BATCH_SIZE_ARGUMENT.default)
            mode = args[REQUIRED_MODE_ARGUMENT.name]
            limit = args.get(OPTIONAL_LIMIT_ARGUMENT.name)

            # Validate mode argument
            if mode not in ["refresh", "resume"]:
                return Result.FAILURE, f"{X} Invalid mode: {mode}. Must be 'refresh' or 'resume'"

            if limit:
                logger.debug("Limit: %d clusters", limit)
            else:
                logger.debug("No limit specified")

            sqlconn = get_sql_conn(namespace)
            topic_discovery = TopicDiscovery.get_from_env()
            topic_discovery.accumulated_errors = 0

            if mode == "refresh":
                logger.info("In refresh mode: Removing any existing cluster topics and starting fresh")
                remove_existing_cluster_topics(sqlconn, namespace)
            elif mode == "resume":
                logger.debug("In resume mode: resuming progress on existing cluster topics")
                # make sure we have the new columns for tracking adverse topic labeling progress

            # get all the cluster nodes
            logger.debug("Loading all cluster nodes in namespace %s", namespace)
            cluster_node_list = get_all_cluster_nodes_for_topic_labeling(sqlconn, namespace)

            # figure out which are leaf nodes
            logger.debug("Determining leaf nodes")
            parent_node_ids = set(node.parent_id for node in cluster_node_list)

            # label leaf nodes
            leaf_count = 0
            for node in cluster_node_list:
                if node.node_id not in parent_node_ids:
                    node.is_leaf = True
                    leaf_count += 1

            # load page content
            logger.debug("Loading page content for clusters")
            page_content_map = get_pages_in_all_clusters(sqlconn, namespace)

            # map sibling clusters by parent_id
            logger.debug("Mapping parent to child clusters")
            parent_id_child_map = {}
            for node in cluster_node_list:
                if not node.parent_id:
                    continue
                if node.parent_id not in parent_id_child_map:
                    parent_id_child_map[node.parent_id] = list()
                parent_id_child_map[node.parent_id].append(node)

            # report some stats
            cluster_count = len(cluster_node_list)
            logger.debug("Cluster count: %d", cluster_count)
            logger.debug("Leaf count: %d", leaf_count)
            logger.debug("Stem count: %d", cluster_count - leaf_count)

            # get a leaf node work list
            if mode == "refresh":
                leaf_node_pass_1_work_list = [node for node in cluster_node_list if node.is_leaf]
            else:
                leaf_node_pass_1_work_list = [node for node in cluster_node_list
                                              if node.is_leaf and not node.first_label]

            # Track total clusters processed across all passes
            total_clusters_processed = 0

            # Pass 1: generation of leaf cluster node topics
            if leaf_node_pass_1_work_list:
                work_list_length = len(leaf_node_pass_1_work_list)
                logger.debug("Executing pass 1: naive generation of leaf cluster node topics")
                logger.debug("leaf_node_pass_1_work_list count: %d", work_list_length)
                with ProgressTracker(description="Pass 1/4: Naive leaf node topics", unit="cluster",
                                     total=work_list_length) as tracker:
                    for batch in batched(leaf_node_pass_1_work_list, batch_size):
                        if limit and total_clusters_processed >= limit:
                            logger.debug("Reached processing limit")
                            break
                        if limit and len(batch) > limit - total_clusters_processed:
                            logger.debug("Truncating batch size to %d because of limit",
                                         limit - total_clusters_processed)
                            batch = batch[:limit - total_clusters_processed]
                        topic_discovery.naively_summarize_page_topics_batch(namespace, batch, page_content_map)
                        update_cluster_tree_first_labels(sqlconn, namespace, batch)
                        tracker.update(len(batch))
                        total_clusters_processed += len(batch)
                        logger.debug("After batch, total processed is now %d", total_clusters_processed)
            else:
                logger.debug("Skipping pass 1 because no nodes need work")

            if limit and total_clusters_processed >= limit:
                return (Result.SUCCESS, f"{CHECK} Stopping processing at limit in pass 1. "
                        f"Total clusters processed for namespace {namespace}: {total_clusters_processed}.")
            logger.debug("Total processed after pass 1: %d", total_clusters_processed)

            # Pass 2: adversarial re-generation of leaf cluster node topics

            # get a leaf node work list
            if mode == "refresh":
                leaf_node_pass_2_work_list = [node for node in cluster_node_list if node.is_leaf]
            else:
                leaf_node_pass_2_work_list = [node for node in cluster_node_list
                                              if node.is_leaf and not node.final_label]

            if leaf_node_pass_2_work_list:
                work_list_length = len(leaf_node_pass_2_work_list)
                logger.debug("Executing pass 2: adversarial re-generation of leaf cluster node topics")
                logger.debug("leaf_node_pass_2_work_list count: %d", work_list_length)
                with ProgressTracker(description="Pass 2/4: Adversarial leaf node topics", unit="cluster",
                                     total=work_list_length) as tracker:

                    for batch in batched(leaf_node_pass_2_work_list, batch_size):
                        if limit and total_clusters_processed >= limit:
                            logger.debug("Reached processing limit")
                            break
                        if limit and len(batch) > limit - total_clusters_processed:
                            logger.debug("Truncating batch size to %d because of limit",
                                         limit - total_clusters_processed)
                            batch = batch[:limit - total_clusters_processed]
                        topic_discovery.adversely_summarize_page_topics_batch(
                            namespace, batch, page_content_map, parent_id_child_map,
                        )
                        update_cluster_tree_final_labels(sqlconn, namespace, batch)
                        tracker.update(len(batch))
                        total_clusters_processed += len(batch)
                        logger.debug("After batch, total processed is now %d", total_clusters_processed)

            else:
                logger.debug("Skipping pass 2 because no nodes need work")

            if limit and total_clusters_processed >= limit:
                return (Result.SUCCESS, f"{CHECK} Stopping processing at limit in pass 2. "
                        f"Total clusters processed for namespace {namespace}: {total_clusters_processed}.")
            logger.debug("Total processed after pass 1: %d", total_clusters_processed)

            # Pass 3: Label topics for nodes that don't have topics yet. Should be non-leaf nodes
            pass_3_work_list = [
                node for node in cluster_node_list if not node.first_label and node.parent_id and not node.is_leaf
            ]
            logger.debug("Sorting pass 3 worklist to go from higher depth to lower depth")
            pass_3_work_list = sorted(pass_3_work_list, reverse=True, key=lambda node: node.depth)

            # validate that the sorting worked as expected
            # TODO delete or comment out this later once we are convinced that it works
            for previous, current in zip(pass_3_work_list, pass_3_work_list[1:]):
                assert previous.depth >= current.depth, "Depth sorting did not work"

            if pass_3_work_list:
                work_list_length = len(pass_3_work_list)
                logger.debug("Executing pass 3: naive generation of stem cluster node topics")
                logger.debug("pass_3_work_list count: %d", work_list_length)
                with ProgressTracker(description="Pass 3/4: Naive stem topics",
                                     unit="cluster",
                                     total=work_list_length) as tracker:
                    for batch in batched(pass_3_work_list, batch_size):
                        if limit and total_clusters_processed >= limit:
                            logger.debug("Reached processing limit")
                            break
                        if limit and len(batch) > limit - total_clusters_processed:
                            logger.debug("Truncating batch size to %d because of limit",
                                         limit - total_clusters_processed)
                            batch = batch[:limit - total_clusters_processed]
                        topic_discovery.naively_summarize_cluster_topics_batch(
                            namespace, batch, parent_id_child_map
                        )
                        update_cluster_tree_first_labels(sqlconn, namespace, batch)
                        tracker.update(len(batch))
                        total_clusters_processed += len(batch)
                        logger.debug("After batch, total processed is now %d", total_clusters_processed)
            else:
                logger.debug("Skipping pass 3 because no nodes need work")

            if limit and total_clusters_processed >= limit:
                return (Result.SUCCESS, f"{CHECK} Stopping processing at limit in pass 3. "
                        f"Total clusters processed for namespace {namespace}: {total_clusters_processed}.")
            logger.debug("Total processed after pass 3: %d", total_clusters_processed)

            # Pass 4: Adversarial label for non-leaf nodes
            pass_4_work_list = pass_3_work_list.copy()
            logger.debug("Sorting pass 4 worklist to go from higher depth to lower depth")
            pass_4_work_list = sorted(pass_4_work_list, reverse=True, key=lambda node: node.depth)

            # validate that the sorting worked as expected
            # TODO delete or comment out this later once we are convinced that it works
            for previous, current in zip(pass_4_work_list, pass_4_work_list[1:]):
                assert previous.depth >= current.depth, "Depth sorting did not work"

            if pass_4_work_list:  # Only run pass 4 if pass 3 had clusters to process
                work_list_length = len(pass_4_work_list)
                logger.debug("Executing pass 4: adversarial re-generation of stem cluster node topics")
                logger.debug("nodes_missing_topics count: %d", work_list_length)
                with ProgressTracker(description="Pass 4/4: Adversarial stem topics",
                                     unit="cluster",
                                     total=work_list_length) as tracker:
                    for batch in batched(pass_4_work_list, batch_size):
                        if limit and total_clusters_processed >= limit:
                            logger.debug("Reached processing limit")
                            break
                        if limit and len(batch) > limit - total_clusters_processed:
                            logger.debug("Truncating batch size to %d because of limit",
                                         limit - total_clusters_processed)
                            batch = batch[:limit - total_clusters_processed]

                        nodes_with_neighbors = [n for n in batch
                                                if parent_id_child_map[n.parent_id]
                                                and len(parent_id_child_map[n.parent_id]) > 1]
                        nodes_without_neighbors = [n for n in batch if n not in nodes_with_neighbors]

                        topic_discovery.adversely_summarize_cluster_topics_batch(
                            namespace, nodes_with_neighbors, parent_id_child_map
                        )
                        for node in nodes_without_neighbors:
                            node.final_label = node.first_label

                        update_cluster_tree_final_labels(sqlconn, namespace, nodes_with_neighbors)
                        update_cluster_tree_final_labels(sqlconn, namespace, nodes_without_neighbors)

                        tracker.update(len(batch))
                        total_clusters_processed += len(batch)
                        logger.debug("After batch, total processed is now %d", total_clusters_processed)
            else:
                logger.debug("Skipping pass 4 because no nodes need work")

            if limit and total_clusters_processed >= limit:
                return (Result.SUCCESS, f"{CHECK} Stopping processing at limit in pass 4. "
                        f"Total clusters processed for namespace {namespace}: {total_clusters_processed}.")
            logger.debug("Total processed after pass 4: %d", total_clusters_processed)

            # TODO write "wikipedia" in the language appropriate way
            root_label = f"{get_language_for_namespace(namespace)} Wikipedia"
            root_node = [n for n in cluster_node_list if not n.parent_id][0]
            root_node.first_label = root_node.final_label = root_label
            logger.info("Root node ID: %s", root_node.node_id)
            logger.info("Root node label: %s", root_label)
            update_cluster_tree_first_labels(sqlconn, namespace, [root_node])
            update_cluster_tree_final_labels(sqlconn, namespace, [root_node])

            logger.info("Handled %d errors from openai endpoint", topic_discovery.accumulated_errors)
            logger.info("Total clusters processed across all passes: %d", total_clusters_processed)

            return Result.SUCCESS, f"{CHECK} Completed cluster topic discovery for namespace {namespace}."

        except Exception as e:
            logger.exception(f"Failed to discover cluster topics: {e}")
            return Result.FAILURE, f"{X} Failed to discover cluster topics: {e}"


class ProjectCommand(Command):
    """Project reduced vector clusters into 3-space."""

    def __init__(self):
        super().__init__(
            name="project",
            description="Project reduced vector clusters into 3-space.",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT, OPTIONAL_CLUSTER_LIMIT_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
            limit = args.get(OPTIONAL_CLUSTER_LIMIT_ARGUMENT.name)

            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            with ProgressTracker(description="Projecting into 3-space", unit="vectors") as tracker:
                processed_count = run_umap_per_cluster(sqlconn, namespace, n_components=3, limit=limit, tracker=tracker)

            return (
                Result.SUCCESS,
                f"{CHECK} Projected reduced page embeddings in {namespace} in {processed_count} "
                f"cluster{'' if processed_count == 1 else 's'} into 3-space using UMAP."
            )
        except Exception as e:
            logger.exception(f"Failed to cluster embeddings: {e}")
            return Result.FAILURE, f"{X} Failed to project reduced page embeddings into 3-space: {e}"


class StatusCommand(Command):
    """Show current data status."""

    def __init__(self):
        super().__init__(
            name="status",
            description="Show current data status",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
            sqlconn = get_sql_conn(namespace)
            ensure_tables(sqlconn)

            # get distinct namespaces
            namespaces_sql = """
                SELECT DISTINCT namespace
                FROM chunk_log
                ORDER BY namespace ASC;
            """
            cursor = sqlconn.execute(namespaces_sql)
            namespace_list = [row[0] for row in cursor.fetchall()]

            # iterate namespaces
            ns_index = 0
            for namespace in namespace_list:
                ns_index += 1
                print(f"\nNamespace {namespace} ({ns_index}/{len(namespace_list)})\n")

                # count and show chunk status
                chunk_cursor = sqlconn.execute(
                    """
                    SELECT
                        COUNT(*) as total_chunks,
                        SUM(CASE WHEN downloaded_at IS NOT NULL THEN 1 ELSE 0 END) as downloaded_chunks,
                        SUM(CASE WHEN chunk_extracted_path IS NOT NULL THEN 1 ELSE 0 END) as extracted_chunks
                    FROM chunk_log
                    WHERE namespace = ?
                    """, (namespace, )
                )
                chunk_stats = chunk_cursor.fetchone()

                print(f"    Chunks: "
                      f"{chunk_stats['downloaded_chunks']} downloaded, "
                      f"{chunk_stats['extracted_chunks']} extracted, "
                      f"{chunk_stats['total_chunks']} total"
                      )

                # count and show page stats
                page_log_stats_sql = """
                    SELECT COUNT(*)
                    FROM page_log
                    WHERE namespace = ?
                    AND page_id IS NOT NULL
                """
                page_log_stats_cursor = sqlconn.execute(page_log_stats_sql, (namespace, ))
                page_log_stats = page_log_stats_cursor.fetchone()

                page_vector_stats_sql = """
                    SELECT
                        SUM(CASE WHEN embedding_vector IS NOT NULL THEN 1 ELSE 0 END) AS embedding_count,
                        SUM(CASE WHEN reduced_vector IS NOT NULL THEN 1 ELSE 0 END) as reduced_count,
                        SUM(CASE WHEN three_d_vector IS NOT NULL THEN 1 ELSE 0 END) as three_d_count,
                        SUM(CASE WHEN cluster_id IS NOT NULL THEN 1 ELSE 0 END) as clustered_count,
                        SUM(CASE WHEN cluster_node_id IS NOT NULL THEN 1 ELSE 0 END) as tree_clustered_count
                    FROM page_vector
                    WHERE namespace = ?
                """
                page_vector_stats_cursor = sqlconn.execute(page_vector_stats_sql, (namespace, ))
                page_vector_stats = page_vector_stats_cursor.fetchone()

                print(f"    Pages: "
                      f"{page_vector_stats['embedding_count']} embeddings, "
                      f"{page_vector_stats['reduced_count']} reduced, "
                      f"{page_vector_stats['three_d_count']} 3D projections, "
                      f"{page_vector_stats['clustered_count']} clustered, "
                      f"{page_vector_stats['tree_clustered_count']} tree-clustered, "
                      f"{page_log_stats[0]} total"
                      )

                # count and show cluster stats
                tree_cluster_count_sql = """
                    SELECT COUNT(*) as the_count,
                        SUM(CASE WHEN first_label IS NOT NULL THEN 1 ELSE 0 END) as first_label_count,
                        SUM(CASE WHEN final_label IS NOT NULL THEN 1 ELSE 0 END) as final_label_count
                    FROM cluster_tree
                    WHERE namespace = ?;
                """
                tree_cluster_count_cursor = sqlconn.execute(tree_cluster_count_sql, (namespace, ))
                tree_cluster_count_stats = tree_cluster_count_cursor.fetchone()

                print(f"    Tree clusters: "
                      f"{tree_cluster_count_stats['first_label_count']} first labeled, "
                      f"{tree_cluster_count_stats['final_label_count']} final labeled, "
                      f"{tree_cluster_count_stats['the_count']} total tree nodes."
                      "\n"
                      )

                tree_depth_stats_sql = """
                    SELECT depth, COUNT() as the_count
                    FROM cluster_tree
                    WHERE namespace = ?
                    GROUP BY depth
                    ORDER BY depth ASC;
                """
                tree_depth_stats_cursor = sqlconn.execute(tree_depth_stats_sql, (namespace, ))
                tree_depth_stats = tree_depth_stats_cursor.fetchall()
                print("    Depth counts:")
                for row in tree_depth_stats:
                    print(f"      {row['depth']: >2}    {row['the_count']: >6} nodes")

                # pad the end of the namespace stuff
                print("")

            return Result.SUCCESS, "Status calculations completed"

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return Result.FAILURE, f"{X} Failed to get status: {e}"


COMMAND_NAME_ARGUMENT = Argument(name="command", type="string", required=False,
                                 description="Get help on the specific named command")


class HelpCommand(Command):
    """Show help information."""

    def __init__(self, parser: CommandParser):
        super().__init__(
            name="help",
            description="Show help information",
            expected_args=[COMMAND_NAME_ARGUMENT]
        )
        self.parser = parser

    def execute(self, args: Dict[str, Any]) -> tuple[Result, str]:
        command_name = args.get(COMMAND_NAME_ARGUMENT.name)

        if command_name:
            return self.parser.get_command_help(command_name)
        else:
            help_text = "Available commands:\n"
            for cmd_name in self.parser.get_available_commands():
                cmd = self.parser.commands[cmd_name]
                help_text += f"  {cmd_name} - {cmd.description}\n"

            help_text += (
                "\nUse 'help <command>' for more information about a specific command."
            )
            return Result.SUCCESS, help_text


HAPPY_PROMPT = "> "


class CommandInterpreter:
    """Main command interpreter class."""

    def __init__(self):
        self.parser = CommandParser()
        self.dispatcher = CommandDispatcher(self.parser)
        self._register_commands()

    def _register_commands(self):
        """Register all commands."""
        self.parser.register_command(RefreshChunkDataCommand())
        self.parser.register_command(DownloadChunksCommand())
        self.parser.register_command(UnpackProcessChunksCommand())
        self.parser.register_command(EmbedPagesCommand())
        self.parser.register_command(ReduceCommand())
        self.parser.register_command(RecursiveClusterCommand())
        self.parser.register_command(ProjectCommand())
        self.parser.register_command(TopicsCommand())
        self.parser.register_command(StatusCommand())
        self.parser.register_command(HelpCommand(self.parser))

    def run_interactive(self):
        """Run interactive command interpreter."""
        print("Welcome to wp-embeddings command interpreter!")
        print("Type 'help' for available commands or 'quit' to exit.")
        print()

        while True:
            try:
                user_input = input(HAPPY_PROMPT).strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                command_name, args = self.parser.parse_input(user_input)

                if not command_name:
                    continue

                result = self.dispatcher.dispatch(command_name, args)
                print(result[1])
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Interactive mode error: {e}")

    def run_command(self, command_args: List[str]) -> Result:
        """Run a single command."""
        if not command_args:
            print("Usage: python command.py <command> [options]")
            return Result.SUCCESS

        command_name = command_args[0]

        # Special handling for help command
        if command_name == "help":
            # Parse help command manually
            help_args = {}
            if len(command_args) > 1:
                help_args["command"] = command_args[1]

            help_command = self.parser.commands.get("help")
            if help_command:
                result = help_command.execute(help_args)
                print(result[1])
                return result[0]
            else:
                return Result.FAILURE

        args = {}

        # Parse command line arguments
        parser = argparse.ArgumentParser(description=f"Execute {command_name} command")

        # Add command-specific arguments
        command = self.parser.commands.get(command_name)
        if command:
            for arg in command.get_required_args():
                if arg.type == "integer":
                    parser.add_argument(
                        f"--{arg.name}", type=int, required=True, help=f"Required argument: {arg.name}"
                    )
                elif arg.type == "float":
                    parser.add_argument(
                        f"--{arg.name}", type=float, required=True, help=f"Required argument: {arg.name}"
                    )
                else:
                    parser.add_argument(
                        f"--{arg.name}", required=True, help=f"Required argument: {arg.name}"
                    )

            for arg in command.get_optional_args():
                if arg.type == "integer":
                    parser.add_argument(
                        f"--{arg.name}", type=int, default=arg.default, help=f"Optional argument: {arg.name}"
                    )
                elif arg.type == "float":
                    parser.add_argument(
                        f"--{arg.name}", type=float, default=arg.default, help=f"Optional argument: {arg.name}"
                    )
                else:
                    parser.add_argument(
                        f"--{arg.name}", default=arg.default, help=f"Optional argument: {arg.name}"
                    )

        try:
            parsed_args = parser.parse_args(command_args[1:])
            args = vars(parsed_args)
        except SystemExit:
            return Result.FAILURE

        result = self.dispatcher.dispatch(command_name, args)
        print(result[1])
        return result[0]


def main() -> int:
    """Main entry point."""
    interpreter = CommandInterpreter()

    if len(sys.argv) > 1:
        # Run single command
        result = interpreter.run_command(sys.argv[1:])
        return result
    else:
        # Run interactive mode
        interpreter.run_interactive()
        return Result.SUCCESS.value


if __name__ == "__main__":
    sys.exit(main())
