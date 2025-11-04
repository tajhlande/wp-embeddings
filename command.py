import argparse
import atexit
import json
import logging
import os
import readline
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any

# Import existing modules
from database import (
    get_embedding_count,
    get_reduced_vector_count,
    get_sql_conn,
    ensure_tables,
    get_page_ids_needing_embedding_for_chunk,
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
from progress_utils import ProgressTracker
from transform import run_kmeans, run_pca, run_umap_per_cluster

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


REQUIRED_NAMESPACE_ARGUMENT = Argument(name="namespace", type="string", required=True,
                                       description="The wiki namespace, e.g. enwiki_namespace_0")
OPTIONAL_NAMESPACE_ARGUMENT = Argument(name="namespace", type="string", required=False,
                                       description="The wiki namespace, e.g. enwiki_namespace_0")
OPTIONAL_CHUNK_LIMIT_ARGUMENT = Argument(name="limit", type="integer", required=False, default=1,
                                         description="Number of chunks to process")
OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT = Argument(name="limit", type="integer", required=False,
                                                    description="Number of chunks to process")
OPTIONAL_CHUNK_NAME_ARGUMENT = Argument(name="chunk", type="string", required=False,
                                        description="The name of the chunk to process")
OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT = Argument(name="limit", type="integer", required=False,
                                                   description="Number of pages to process")


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
    def execute(self, args: Dict[str, Any]) -> str:
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
                    if val < 1:
                        raise ValueError(f"Argument {expected_arg.name} must be >= 1")
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

    def get_command_help(self, command_name: str) -> str:
        """Get help for a specific command."""
        if command_name in self.commands:
            return self.commands[command_name].get_help()
        return f"Unknown command: {command_name}"

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

    def dispatch(self, command_name: str, args: Dict[str, Any]) -> str:
        """Dispatch command to appropriate handler."""
        if not self.parser.validate_command(command_name):
            return f"Unknown command: {command_name}"

        try:
            command = self.parser.commands[command_name]
            command.validate(args)
            return command.execute(args)
        except ValueError as e:
            return f"Validation error: {e}"
        except Exception as e:
            logger.exception(f"Error executing command '{command_name}': {e}")
            return f"Error: {e}"


class RefreshChunkDataCommand(Command):
    """Refresh chunk data for a namespace."""

    def __init__(self):
        super().__init__(
            name="refresh",
            description="Refresh chunk data for a namespace",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
        logger.info("Refreshing chunk data for namespace: %s", namespace)

        try:
            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            api_client = self._get_api_client()
            get_chunk_info_for_namespace(namespace, api_client, sqlconn)

            # Get count of chunks for the namespace
            cursor = sqlconn.execute(
                "SELECT COUNT(*) as count FROM chunk_log WHERE namespace = ?",
                (namespace,),
            )
            count = cursor.fetchone()["count"]

            return f"✓ Refreshed chunk data for namespace: {namespace}\nFound {count} chunks in namespace {namespace}"

        except Exception as e:
            logger.error(f"Failed to refresh chunk data: {e}")
            return f"✗ Failed to refresh chunk data: {e}"

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
            expected_args=[OPTIONAL_CHUNK_LIMIT_ARGUMENT, OPTIONAL_NAMESPACE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        limit = args.get(OPTIONAL_CHUNK_LIMIT_ARGUMENT.name, OPTIONAL_CHUNK_LIMIT_ARGUMENT.default)
        namespace = args.get(OPTIONAL_NAMESPACE_ARGUMENT.name)

        try:
            sqlconn = get_sql_conn()
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
                return "✓ No chunks available for download"

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

            return f"✓ Downloaded {downloaded_count} chunk(s) in namespace {namespace or 'all'}"

        except Exception as e:
            logger.error(f"Failed to download chunks: {e}")
            return f"✗ Failed to download chunks: {e}"

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
            expected_args=[OPTIONAL_NAMESPACE_ARGUMENT, OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        namespace = args.get(OPTIONAL_NAMESPACE_ARGUMENT.name)
        limit = args.get(OPTIONAL_CHUNK_LIMIT_NO_DEFAULT_ARGUMENT.name)

        try:
            sqlconn = get_sql_conn()
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
                return "✓ No chunks available for unpacking"

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
                                sqlconn, chunk_name, chunk_file_path, tracker
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
                f"✓ Unpacked and processed {processed_count} chunk(s). "
                f"Processed {total_pages} pages from {processed_count} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to unpack chunks: {e}")
            return f"✗ Failed to unpack chunks: {e}"


class EmbedPagesCommand(Command):
    """Process remaining pages for embedding computation."""

    def __init__(self):
        super().__init__(
            name="embed",
            description="Process remaining pages for embedding computation",
            expected_args=[OPTIONAL_CHUNK_NAME_ARGUMENT, OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT]
        )
        embedding_model_name, embedding_model_api_url, embedding_model_api_key = (
            get_embedding_model_config()
        )
        self.embedding_model_name = embedding_model_name
        self.embedding_model_api_url = embedding_model_api_url
        self.embedding_model_api_key = embedding_model_api_key

    def execute(self, args: Dict[str, Any]) -> str:
        chunk_name = args.get(OPTIONAL_CHUNK_NAME_ARGUMENT.name)
        limit = args.get(OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT.name, OPTIONAL_PAGE_LIMIT_NO_DEFAULT_ARGUMENT.default)

        logger.info(
            "Computing embeddings for pages%s%s...",
            f" in chunk {chunk_name}" if chunk_name else "",
            f" with limit {limit}" if limit else "",
        )

        try:
            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            # Setup embedding function
            embedding_function = get_embedding_function(
                model_name=self.embedding_model_name,
                openai_compatible_url=self.embedding_model_api_url,
                openai_api_key=self.embedding_model_api_key,
            )

            if chunk_name:
                # Process specific chunk
                page_ids = get_page_ids_needing_embedding_for_chunk(chunk_name, sqlconn)

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
                    return f"✓ No pages needing embeddings in chunk {chunk_name}"

                # logger.info("Processing %d embeddings for named chunk %s...", len(page_ids), chunk_name)

                # Compute embeddings
                with ProgressTracker(
                    f"Computing embeddings for chunk {chunk_name}",
                    total=pages_to_process,
                    unit="pages",
                ) as tracker:
                    compute_embeddings_for_chunk(
                        chunk_name,
                        embedding_function,
                        sqlconn,
                        limit=pages_to_process,
                        tracker=tracker,
                    )

                return f"✓ Processed {len(page_ids)} pages for chunk {chunk_name}"

            else:
                # Process all chunks
                logger.debug(
                    "Processing %sembeddings for next available chunk...",
                    f"up to {limit} " if limit else "",
                )

                cursor = sqlconn.execute(
                    """
                    SELECT DISTINCT chunk_name
                    FROM page_log
                    LEFT JOIN page_vector ON page_log.page_id = page_vector.page_id
                    WHERE page_vector.embedding_vector IS NULL;
                    """
                )
                chunk_name_list = [row["chunk_name"] for row in cursor.fetchall()]

                if not chunk_name_list:
                    return "✓ No pages needing embeddings"

                total_processed = 0

                for chunk_name in chunk_name_list:
                    try:
                        # Check if we have remaining limit capacity
                        if limit and total_processed >= limit:
                            break

                        # Get pages needing embeddings for this chunk
                        page_ids = get_page_ids_needing_embedding_for_chunk(
                            chunk_name, sqlconn
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
                                    chunk_name,
                                    embedding_function,
                                    sqlconn,
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

                return f"✓ Embedded {total_processed} pages across {len(chunk_name_list)} chunk(s)"

        except Exception as e:
            logger.exception(f"Failed to process pages: {e}")
            return f"✗ Failed to process pages: {e}"


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

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]

            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            target_dim = args.get(TARGET_DIMENSIONS_ARGUMENT.name, TARGET_DIMENSIONS_ARGUMENT.default)
            batch_size = args.get(BATCH_SIZE_ARGUMENT.name, BATCH_SIZE_ARGUMENT.default)
            print(f"Target dimension: {target_dim}, batch size: {batch_size}")

            if target_dim > batch_size:
                return "✗ Batch size must be equal to or greater than target dimension."

            estimated_vector_count = get_embedding_count(namespace, sqlconn)
            estimated_batch_count = estimated_vector_count // batch_size + 1

            if estimated_vector_count < target_dim:
                return (
                    f"✗ Only found {estimated_vector_count} records, "
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
                unit="batches",
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
                f"✓ Reduced {total_vector_count} page embeddings in {batch_count} "
                f"batch{'' if batch_count == 1 else 'es'}"
            )
        except Exception as e:
            logger.exception(f"Failed to reduce embeddings: {e}")
            return f"✗ Failed to reduce embeddings: {e}"


CLUSTER_COUNT_ARGUMENT = Argument(name="clusters", type="integer", required=False, default=10_000,
                                  description="Number of clusters to process")


class ClusterCommand(Command):
    """Cluster reduced vectors with k-means."""

    def __init__(self):
        super().__init__(
            name="cluster",
            description="Cluster reduced vectors with k-means",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT, CLUSTER_COUNT_ARGUMENT, BATCH_SIZE_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]

            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            num_clusters = args.get(CLUSTER_COUNT_ARGUMENT.name, CLUSTER_COUNT_ARGUMENT.default)
            batch_size = args.get(BATCH_SIZE_ARGUMENT.name, BATCH_SIZE_ARGUMENT.default)

            estimated_vector_count = get_reduced_vector_count(namespace, sqlconn)
            estimated_batch_count = estimated_vector_count // batch_size + 1

            if estimated_vector_count < batch_size:
                print(
                    f"Reducing batch size to match estimated vector count: {estimated_vector_count}"
                )
                batch_size = estimated_vector_count

            print(f"Clustering to {num_clusters} clusters in batches of {batch_size} in namespace {namespace}")

            with ProgressTracker(
                "K-means Clustering (two pass)",
                unit="batches",
                total=estimated_batch_count * 2,
            ) as tracker:
                run_kmeans(
                    sqlconn,
                    namespace=namespace,
                    n_clusters=num_clusters,
                    batch_size=batch_size,
                    tracker=tracker,
                )

            return f"✓ Clustered reduced page embeddings in namespace {namespace} using incremental K-means"
        except Exception as e:
            logger.error(f"Failed to cluster embeddings: {e}")
            return f"✗ Failed to cluster embeddings: {e}"


OPTIONAL_CLUSTER_LIMIT_ARGUMENT = Argument(name="limit", type="integer", required=False,
                                           description="Number of clusters to process")


class ProjectCommand(Command):
    """Project reduced vector clusters into 3-space."""

    def __init__(self):
        super().__init__(
            name="project",
            description="Project reduced vector clusters into 3-space.",
            expected_args=[REQUIRED_NAMESPACE_ARGUMENT, OPTIONAL_CLUSTER_LIMIT_ARGUMENT]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            namespace = args[REQUIRED_NAMESPACE_ARGUMENT.name]
            limit = args.get(OPTIONAL_CLUSTER_LIMIT_ARGUMENT.name)

            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            with ProgressTracker(description="Projecting into 3-space", unit="vectors") as tracker:
                processed_count = run_umap_per_cluster(sqlconn, namespace, n_components=3, limit=limit, tracker=tracker)

            return f"✓ Projected reduced page embeddings in {namespace} in {processed_count} " \
                   f"cluster{'' if processed_count == 1 else 's'} into 3-space using UMAP."
        except Exception as e:
            logger.exception(f"Failed to cluster embeddings: {e}")
            return f"✗ Failed to project reduced page embeddings into 3-space: {e}"


class StatusCommand(Command):
    """Show current data status."""

    def __init__(self):
        super().__init__(
            name="status",
            description="Show current data status",
            expected_args=[]
        )

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            sqlconn = get_sql_conn()
            ensure_tables(sqlconn)

            # Get chunk statistics
            chunk_cursor = sqlconn.execute(
                """
                SELECT
                    COUNT(*) as total_chunks,
                    SUM(CASE WHEN downloaded_at IS NOT NULL THEN 1 ELSE 0 END) as downloaded_chunks,
                    SUM(CASE WHEN chunk_extracted_path IS NOT NULL THEN 1 ELSE 0 END) as extracted_chunks
                FROM chunk_log
            """
            )
            chunk_stats = chunk_cursor.fetchone()

            chunk_completion_cursor = sqlconn.execute(
                """
                SELECT
                       SUM(CASE
                           WHEN total_pages > 0 AND pending_embeddings = 0
                           THEN 1
                           ELSE 0 END) as "chunks_completed_embedding",
                       SUM(CASE
                           WHEN total_pages = 0 OR pending_embeddings > 0
                           THEN 1
                           ELSE 0 END) as "chunks_pending_embedding",
                       SUM(pending_embeddings) as "pending_embeddings_count",
                       SUM(completed_embeddings) as "completed_embeddings_count",

                       SUM(CASE
                           WHEN total_pages > 0 AND pending_reduced = 0
                           THEN 1
                           ELSE 0 END) as "chunks_completed_reduction",
                       SUM(CASE
                           WHEN total_pages = 0 OR pending_reduced > 0
                           THEN 1
                           ELSE 0 END) as "chunks_pending_reduction",
                       SUM(pending_reduced) as "pending_reduced_count",
                       SUM(completed_reduced) as "completed_reduced_count",

                       SUM(CASE
                           WHEN total_pages > 0 AND pending_clustering = 0
                           THEN 1
                           ELSE 0 END) as "chunks_completed_clustering",
                       SUM(CASE
                           WHEN total_pages = 0 OR pending_clustering > 0
                           THEN 1
                           ELSE 0 END) as "chunks_pending_clustering",
                       SUM(pending_clustering) as "pending_clustering_count",
                       SUM(completed_clustering) as "completed_clustering_count",

                       SUM(CASE
                           WHEN total_pages > 0 AND pending_projection = 0
                           THEN 1
                           ELSE 0 END) as "chunks_completed_projection",
                       SUM(CASE
                           WHEN total_pages = 0 OR pending_projection > 0
                           THEN 1
                           ELSE 0 END) as "chunks_pending_projection",
                       SUM(pending_projection) as "pending_projection_count",
                       SUM(completed_projection) as "completed_projection_count",

                       SUM(total_pages) as "total_pages_count"
                    FROM (
                        SELECT chunk_log.chunk_name,
                           SUM(CASE
                               WHEN page_vector.embedding_vector IS NULL and page_log.page_id IS NOT NULL
                               THEN 1
                               ELSE 0 END) as pending_embeddings,
                           SUM(CASE
                               WHEN page_vector.embedding_vector IS NULL
                               THEN 0
                               ELSE 1 END) as completed_embeddings,
                           SUM(CASE
                               WHEN page_vector.reduced_vector IS NULL and page_log.page_id IS NOT NULL
                               THEN 1
                               ELSE 0 END) as pending_reduced,
                           SUM(CASE
                               WHEN page_vector.reduced_vector IS NULL
                               THEN 0
                               ELSE 1 END) as completed_reduced,
                           SUM(CASE
                               WHEN page_vector.cluster_id IS NULL and page_log.page_id IS NOT NULL
                               THEN 1
                               ELSE 0 END) as pending_clustering,
                           SUM(CASE
                               WHEN page_vector.cluster_id IS NULL
                               THEN 0
                               ELSE 1 END) as completed_clustering,
                           SUM(CASE
                               WHEN page_vector.three_d_vector IS NULL and page_log.page_id IS NOT NULL
                               THEN 1
                               ELSE 0 END) as pending_projection,
                           SUM(CASE
                               WHEN page_vector.three_d_vector IS NULL
                               THEN 0
                               ELSE 1 END) as completed_projection,
                           COUNT(page_log.page_id) as total_pages
                        FROM chunk_log
                        LEFT JOIN page_log ON chunk_log.chunk_name = page_log.chunk_name
                        LEFT JOIN page_vector ON page_log.page_id = page_vector.page_id
                        GROUP BY chunk_log.chunk_name
                    );
            """
            )
            chunk_completion_stats = chunk_completion_cursor.fetchone()

            # Get page statistics
            page_stats_sql = """
                SELECT
                    COUNT(*) as total_pages,
                    SUM(CASE WHEN embedding_vector IS NOT NULL THEN 1 ELSE 0 END) as embeddings,
                    SUM(CASE WHEN reduced_vector IS NOT NULL THEN 1 ELSE 0 END) as reduced_vectors,
                    SUM(CASE WHEN cluster_id IS NOT NULL THEN 1 ELSE 0 END) as clustered,
                    SUM(CASE WHEN three_d_vector IS NOT NULL THEN 1 ELSE 0 END) as three_d_vectors
                FROM page_log
                LEFT JOIN page_vector ON page_log.page_id = page_vector.page_id
            """
            page_cursor = sqlconn.execute(page_stats_sql)
            page_stats = page_cursor.fetchone()

            status_text = "Data status:\n"

            # Get namespace breakdown
            namespace_cursor = sqlconn.execute(
                """
                SELECT namespace,
                       COUNT(*) as chunk_count,
                       SUM(CASE WHEN downloaded_at IS NOT NULL THEN 1 ELSE 0 END) as downloaded_count
                FROM chunk_log
                GROUP BY namespace
                ORDER BY namespace
            """
            )

            status_text += "\nNamespaces:\n"
            for row in namespace_cursor.fetchall():
                status_text += f"  {row['namespace']}: {row['chunk_count']} chunks, "
                status_text += f"{row['downloaded_count']} downloaded\n\n"

            status_text += f"Chunks: {chunk_stats['total_chunks']} total, "
            status_text += f"{chunk_stats['downloaded_chunks']} downloaded, "
            status_text += f"{chunk_stats['extracted_chunks']} extracted\n"

            status_text += f"Chunk embedding: {chunk_completion_stats['chunks_completed_embedding']} complete, "
            status_text += f"{chunk_completion_stats['chunks_pending_embedding']} pending, "
            status_text += f"{chunk_completion_stats['pending_embeddings_count']} pages pending, "
            status_text += f"{chunk_completion_stats['completed_embeddings_count']} pages completed\n"

            status_text += f"Chunk reduction: {chunk_completion_stats['chunks_completed_reduction']} complete, "
            status_text += f"{chunk_completion_stats['chunks_pending_reduction']} pending, "
            status_text += f"{chunk_completion_stats['pending_reduced_count']} pages pending, "
            status_text += f"{chunk_completion_stats['completed_reduced_count']} pages completed\n"

            status_text += f"Chunk clustering: {chunk_completion_stats['chunks_completed_clustering']} complete, "
            status_text += f"{chunk_completion_stats['chunks_pending_clustering']} pending, "
            status_text += f"{chunk_completion_stats['pending_clustering_count']} pages pending, "
            status_text += f"{chunk_completion_stats['completed_clustering_count']} pages completed\n"

            status_text += f"Chunk projection: {chunk_completion_stats['chunks_completed_projection']} complete, "
            status_text += f"{chunk_completion_stats['chunks_pending_projection']} pending, "
            status_text += f"{chunk_completion_stats['pending_projection_count']} pages pending, "
            status_text += f"{chunk_completion_stats['completed_projection_count']} pages completed\n\n"

            status_text += f"Pages: {page_stats['total_pages']} total, "
            status_text += f"{page_stats['embeddings']} embeddings, "
            status_text += f"{page_stats['reduced_vectors']} reduced vectors, "
            status_text += f"{page_stats['clustered']} clustered pages, "
            status_text += f"{page_stats['clustered']} projected vectors\n"

            return status_text

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return f"✗ Failed to get status: {e}"


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

    def execute(self, args: Dict[str, Any]) -> str:
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
            return help_text


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
        self.parser.register_command(ClusterCommand())
        self.parser.register_command(ProjectCommand())
        self.parser.register_command(StatusCommand())
        self.parser.register_command(HelpCommand(self.parser))

    def run_interactive(self):
        """Run interactive command interpreter."""
        print("Welcome to wp-embeddings command interpreter!")
        print("Type 'help' for available commands or 'quit' to exit.")
        print()

        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                command_name, args = self.parser.parse_input(user_input)

                if not command_name:
                    continue

                result = self.dispatcher.dispatch(command_name, args)
                print(result)
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

    def run_command(self, command_args: List[str]):
        """Run a single command."""
        if not command_args:
            print("Usage: python command.py <command> [options]")
            return

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
                print(result)
            return

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
                else:
                    parser.add_argument(
                        f"--{arg.name}", required=True, help=f"Required argument: {arg.name}"
                    )

            for arg in command.get_optional_args():
                if arg.type == "integer":
                    parser.add_argument(
                        f"--{arg.name}", type=int, default=arg.default, help=f"Optional argument: {arg.name}"
                    )
                else:
                    parser.add_argument(
                        f"--{arg.name}", default=arg.default, help=f"Optional argument: {arg.name}"
                    )

        try:
            parsed_args = parser.parse_args(command_args[1:])
            args = vars(parsed_args)
        except SystemExit:
            return

        result = self.dispatcher.dispatch(command_name, args)
        print(result)


def main():
    """Main entry point."""
    interpreter = CommandInterpreter()

    if len(sys.argv) > 1:
        # Run single command
        interpreter.run_command(sys.argv[1:])
    else:
        # Run interactive mode
        interpreter.run_interactive()


if __name__ == "__main__":
    main()
