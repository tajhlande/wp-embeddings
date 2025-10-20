# Command Interpreter Implementation Plan

## Overview
This document outlines the implementation plan for a command interpreter for the wp-snapshot-dl project. The interpreter will provide four main commands for managing chunk data downloading, processing, and embedding computation.

## Architecture Design

### Core Components

#### 1. Command Base Class
- **Purpose**: Abstract base class for all commands
- **Methods**:
  - `execute()`: Main execution method
  - `validate()`: Input validation
  - `get_help()`: Help text display
- **Properties**:
  - `name`: Command name
  - `description`: Command description
  - `required_args`: List of required arguments
  - `optional_args`: Dictionary of optional arguments with defaults

#### 2. Command Parser
- **Purpose**: Parse user input into command + arguments
- **Features**:
  - Command aliases and shorthand support
  - Parameter validation
  - Help system integration
- **Methods**:
  - `parse_input()`: Parse raw user input
  - `validate_command()`: Check if command exists
  - `get_command_help()`: Get help for specific command

#### 3. Command Dispatcher
- **Purpose**: Route parsed commands to appropriate command classes
- **Features**:
  - Command registration system
  - Error handling and reporting
  - Progress tracking
- **Methods**:
  - `register_command()`: Register new command
  - `dispatch()`: Execute command
  - `get_available_commands()`: List all available commands

### Database Integration
- Leverage existing `common.py` functions:
  - `get_sql_conn()`: Database connection management
  - `ensure_tables()`: Table creation
  - `upsert_new_chunk_data()`: Chunk data operations
  - `upsert_new_page_data()`: Page data operations
  - `get_chunk_data()`: Chunk data retrieval
  - `get_page_ids_needing_embedding_for_chunk()`: Page status checking

### API Client Management
- Use existing `download_chunks.py` functions:
  - `get_enterprise_auth_client()`: Authentication
  - `get_enterprise_api_client()`: API client creation
  - `get_chunk_info_for_namespace()`: Chunk metadata retrieval
  - `download_chunk()`: Chunk downloading
  - `extract_single_file_from_tar_gz()`: File extraction
  - `parse_chunk_file()`: Chunk parsing

- Use existing `index_pages.py` functions:
  - `get_embedding_function()`: Embedding function setup
  - `compute_embeddings_for_chunk()`: Embedding computation
  - `compute_page_embeddings()`: Individual page embedding

## Command Implementations

### 1. RefreshChunkDataCommand
**Purpose**: Refresh chunk data for a namespace
**Usage**: `refresh --namespace <namespace>`

**Implementation**:
```python
class RefreshChunkDataCommand(Command):
    def __init__(self):
        super().__init__(
            name="refresh",
            description="Refresh chunk data for a namespace",
            required_args=["namespace"],
            optional_args={}
        )
    
    def execute(self, args):
        namespace = args["namespace"]
        # Get API client
        auth_client, refresh_token, access_token = get_enterprise_auth_client()
        api_client = get_enterprise_api_client(access_token)
        
        # Refresh chunk data
        sqlconn = get_sql_conn()
        get_chunk_info_for_namespace(namespace, api_client, sqlconn)
        
        return f"Refreshed chunk data for namespace: {namespace}"
```

### 2. DownloadChunksCommand
**Purpose**: Download a certain number of chunks for any chunk that has not yet been downloaded
**Usage**: `download --limit <number> [--namespace <namespace>]`

**Implementation**:
```python
class DownloadChunksCommand(Command):
    def __init__(self):
        super().__init__(
            name="download",
            description="Download chunks that haven't been downloaded yet",
            required_args=[],
            optional_args={"limit": 1, "namespace": None}
        )
    
    def execute(self, args):
        limit = args.get("limit", 1)
        namespace = args.get("namespace")
        
        # Get chunks that need downloading
        sqlconn = get_sql_conn()
        # Query for chunks without downloaded_at
        
        # Download each chunk
        for chunk in chunks_to_download:
            # Download chunk using download_chunk()
            # Parse chunk using parse_chunk_file()
            # Update database with download status
        
        return f"Downloaded {len(chunks_downloaded)} chunks"
```

### 3. UnpackProcessChunksCommand
**Purpose**: Unpack and process pages for chunks that haven't been unpacked and processed, removing the unpacked chunk when complete
**Usage**: `unpack [--namespace <namespace>]`

**Implementation**:
```python
class UnpackProcessChunksCommand(Command):
    def __init__(self):
        super().__init__(
            name="unpack",
            description="Unpack and process downloaded chunks",
            required_args=[],
            optional_args={"namespace": None}
        )
    
    def execute(self, args):
        namespace = args.get("namespace")
        
        # Get chunks that need unpacking
        sqlconn = get_sql_conn()
        # Query for chunks with archive_path but no extracted_path
        
        for chunk in chunks_to_unpack:
            # Extract archive using extract_single_file_from_tar_gz()
            # Parse chunk file using parse_chunk_file()
            # Update database with extracted path
            # Remove archive file
        
        return f"Unpacked and processed {len(chunks_processed)} chunks"
```

### 4. ProcessRemainingPagesCommand
**Purpose**: Process remaining pages for chunks where pages haven't been completely processed yet
**Usage**: `process [--chunk <chunk_name>] [--limit <number>]`

**Implementation**:
```python
class ProcessRemainingPagesCommand(Command):
    def __init__(self):
        super().__init__(
            name="process",
            description="Process remaining pages for embedding computation",
            required_args=[],
            optional_args={"chunk": None, "limit": None}
        )
    
    def execute(self, args):
        chunk_name = args.get("chunk")
        limit = args.get("limit")
        
        # Setup embedding function
        embedding_function = get_embedding_function(
            model_name="jina-embeddings-v4-text-matching-GGUF",
            openai_compatible_url='http://llmhost1.internal.tajh.house:8080/v1',
            openai_api_key='no-key-necessary'
        )
        
        # Get pages needing embeddings
        sqlconn = get_sql_conn()
        page_ids = get_page_ids_needing_embedding_for_chunk(chunk_name, sqlconn)
        
        if limit:
            page_ids = page_ids[:limit]
        
        # Compute embeddings
        compute_embeddings_for_chunk(chunk_name, embedding_function, sqlconn)
        
        return f"Processed {len(page_ids)} pages for chunk {chunk_name}"
```

## File Structure

```
command.py
├── Command (base class)
├── CommandParser
├── CommandDispatcher
├── commands/
│   ├── __init__.py
│   ├── refresh_chunk_data.py
│   ├── download_chunks.py
│   ├── unpack_process_chunks.py
│   └── process_remaining_pages.py
└── main.py (entry point)
```

## Error Handling and Logging

### Error Handling Strategy
1. **API Errors**: Handle authentication failures, rate limiting, and network issues
2. **Database Errors**: Handle connection issues, constraint violations, and data corruption
3. **File Operations**: Handle missing files, permission issues, and disk space problems
4. **Validation Errors**: Handle invalid input parameters and malformed data

### Logging Implementation
- Use existing logging configuration from `common.py`
- Add structured logging for command execution
- Include progress tracking for long-running operations
- Log errors with sufficient context for debugging

## User Interface

### Command Line Interface
- Interactive mode with `python command.py`
- Direct command execution with `python command.py <command> [options]`
- Help system: `python command.py help` or `python command.py <command> --help`

### Progress Reporting
- Real-time progress updates for long-running operations
- Summary statistics upon completion
- Error reporting with actionable messages

### Status Display
- Show current state of chunks and pages
- Display processing statistics
- Provide estimated completion times

## Usage Examples

### Interactive Mode
```bash
# Start interactive command interpreter
python command.py

# Inside the interpreter:
> refresh --namespace enwiki_namespace_0
✓ Refreshed chunk data for namespace: enwiki_namespace_0
Found 25 chunks in namespace enwiki_namespace_0

> download --limit 5
✓ Downloaded 3 chunks (2 already downloaded)
Downloaded 3 chunks in namespace enwiki_namespace_0

> unpack
✓ Unpacked and processed 3 chunks
Processed 1,234 pages from 3 chunks

> process --chunk enwiki_namespace_0_chunk_0 --limit 100
✓ Processed 100 pages for chunk enwiki_namespace_0_chunk_0
Completed embedding computation for 100 pages

> status
Chunks: 25 total, 3 downloaded, 3 processed
Pages: 1,234 total, 1,100 with embeddings

> help
Available commands:
  refresh - Refresh chunk data for a namespace
  download - Download chunks that haven't been downloaded yet
  unpack - Unpack and process downloaded chunks
  process - Process remaining pages for embedding computation
  status - Show current system status
  help - Show this help message
  quit - Exit the interpreter

> quit
Goodbye!
```

### Direct Command Execution
```bash
# Refresh chunk data for a namespace
python command.py refresh --namespace enwiki_namespace_0

# Download up to 5 chunks for any namespace
python command.py download --limit 5

# Download chunks for specific namespace
python command.py download --namespace enwiki_namespace_0 --limit 10

# Unpack and process all downloaded chunks
python command.py unpack

# Unpack chunks for specific namespace
python command.py unpack --namespace enwiki_namespace_0

# Process remaining pages for a specific chunk
python command.py process --chunk enwiki_namespace_0_chunk_0

# Process up to 100 pages for a chunk
python command.py process --chunk enwiki_namespace_0_chunk_0 --limit 100

# Show help
python command.py help
python command.py download --help
```

## Testing Strategy

### Unit Tests
- Test individual command classes
- Mock API calls and database operations
- Test argument parsing and validation
- Test error handling scenarios

### Integration Tests
- Test complete command workflows
- Test database state changes
- Test file system operations
- Test API client interactions

### End-to-End Tests
- Test real command execution
- Test progress reporting
- Test error recovery
- Test help system

## Implementation Timeline

1. **Phase 1**: Core infrastructure (Command base class, Parser, Dispatcher)
2. **Phase 2**: Individual command implementations
3. **Phase 3**: Error handling and logging
4. **Phase 4**: User interface and help system
5. **Phase 5**: Testing and documentation

## Dependencies

The command interpreter will leverage existing dependencies:
- `sqlite3`: Database operations
- `chromadb`: Embedding storage and computation
- `wme_sdk`: API client for WordPress/Wikipedia data
- `numpy`: Vector operations
- `json`: Data serialization
- `logging`: Structured logging
- `argparse`: Command line argument parsing (if needed)

## Security Considerations

- Handle API credentials securely
- Validate user input to prevent injection attacks
- Use proper error handling to avoid information leakage
- Implement proper file permissions for downloaded data
- Sanitize file paths to prevent directory traversal

## Performance Considerations

- Use connection pooling for database operations
- Implement batch processing for large datasets
- Add progress tracking for long-running operations
- Optimize memory usage for large chunk files
- Implement proper cleanup of temporary files

This implementation plan provides a comprehensive foundation for building a robust command interpreter that integrates seamlessly with the existing wp-snapshot-dl codebase.