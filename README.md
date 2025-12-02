# Wikipedia Embeddings and Visualization

This project downloads and extracts Wikipedia article page titles and abstracts from [Wikimedia Enterprise](https://enterprise.wikimedia.com/),
then computes embeddings on them and recursively clusters the embeddings, for the purpose of visualizing all the knowledge in Wikipedia.

The goal is to have visualizations for multiple languages – as many as can be supported by the ML models I've selected for this project:

* [jinaai/jina-embeddings-v4-text-matching-GGUF](https://huggingface.co/jinaai/jina-embeddings-v4-text-matching-GGUF) for embeddings
* [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) for topic discovery

## Project functions and structure

The primary module for the project is `command.py`, and it can be invoked interactively:

```
$ python -m command
Welcome to wp-embeddings command interpreter!
Type 'help' for available commands or 'quit' to exit.

>
```

or with command parameters:

```
$ python -m command help
Available commands:
  refresh - Refresh chunk data for a namespace
  download - Download chunks that haven't been downloaded yet
  unpack - Unpack and process downloaded chunks
  embed - Process remaining pages for embedding computation
  reduce - Reduce dimension of embeddings
  recursive-cluster - Run recursive clustering algorithm to build a tree of clusters
  project - Project reduced vector clusters into 3-space.
  topics - Use an LLM to discover topics for clusters according to their page content.
  status - Show current data status
  help - Show help information

Use 'help <command>' for more information about a specific command.

$
```

Available commands are:

* **refresh** - Fetch metadata from the Wikimedia Enterprise API about snapshots available for download
* **download** – Download snapshot chunks
* **unpack** – Unpack and extract article page titles and abstracts from the snapshot chunks
* **embed** – Compute embeddings on the page titles and abstracts
* **reduce** – Reduce the dimensionality of the embeddings
* **recursive-cluster** – Run recursive clustering with k-means to build a tree of clusters
* **project** – Project the single-pass vectors into 3-space
* **topics** – Use an LLM model to discover topics for clusters according to their page content
* **status** – Show current data status
* **help** – Show help information

Most operations require a `--namespace` argument that is expected to be a Wikipedia namespace,
as they are named by the Wikimedia Enterprise API, for example: `enwiki_namespace_0`.

All the page content, metadata, computed embeddings, and cluster information
are stored in a Sqlite 3 database named after the namespace,
for example `enwiki_namespace_0.db`.

A slightly modified copy of the Wikimedia Enterprise Python SDK is in the `wme_sdk` directory.
Their code has its own license, to be found in the [wme_sdk/LICENSE](wme_sdk/LICENSE) file.

The remainder of the project is licensed by the file in [LICENSE](LICENSE).


## Getting started

This project is managed with [uv](https://docs.astral.sh/uv/), the awesome Python package manager from Astral.

First, install `uv` if you haven't already:

```bash
pip3 install uv
```

Then create a virtual environment and activate it:

```bash
uv venv
source .venv/bin/activate
```

Fetch the project dependencies:

```bash
uv sync
```

Run the command engine interactively:
```
$ python -m command
Welcome to wp-embeddings command interpreter!
Type 'help' for available commands or 'quit' to exit.

> help
Available commands:
  refresh - Refresh chunk data for a namespace
  download - Download chunks that haven't been downloaded yet
  unpack - Unpack and process downloaded chunks
  embed - Process remaining pages for embedding computation
  status - Show current system status
  help - Show help information

Use 'help <command>' for more information about a specific command.

> quit
Goodbye!
```

For required and optional parameters to a command, precede them with a double-dash,
whether on the command line or within the command interpreter:

```bash
python -m command refresh --namespace enwiki_namespace_0
```

## Command notes and tips

Most commands that can operate on more than one item accept a `--limit n` parameter
to limit how many operations they perform,
unless the operation can't be done in incremental pieces.
You can use this capability to manage the work done at any given time.

The `download`, `unpack`, and `embed` commands will all try to avoid repeating work that's already been completed, so you can run
them repeatedly with `--limit` to incrementally do the required work over an entire namespace.

The `topics` command also has a `--mode` option to select either `refresh` or `resume`, which
can be used in combination with `--limit` to manage the workload.  This command is typically the most time consuming,
because it invokes the gpt-oss-20b model to discover topics for each tree cluster node.

## Operational notes

* For accessing the Wikimedia Enterprise API and the LLM API endpoints, the code expects a `.env` file containing valid credentials and configuration.
A sample file can be found in [env-example](env-example).
* Downloaded archive files are stored in `./downloaded/{namespace}` and named like `{chunk_name}.tar.gz`.
* This code assumes that each archive contains exactly one chunk file in ndjson format. If Enterprise changes this, the code must be changed.
* Extracted archives are stored in `./extracted/{namespace}` and are deleted after unpacking and parsing completes (because they are about 2GB each!)
* Both download and extract operations will silently overwrite files if the files exist already.
* Make sure you have enough disk space. For reference, the complete English Wikipedia namespace 0 archive (article pages) takes about 133G in .tar.gz form (as measured in October 2025).

## Embeddings

Embeddings are computed with the `jina-embeddings-v4-text-matching-GGUF` embedding model by default.
Model config is provided through environment variables, and the following parameters are needed:

- `EMBEDDING_MODEL_API_URL` – Required: An OpenAI compatible endpoint for model access
- `EMBEDDING_MODEL_API_KEY` - Required: The key for accessing that API
- `EMBEDDING_MODEL_NAME` - Optional: The name of the model in the API. Defaults to `jina-embeddings-v4-text-matching-GGUF` if not provided

The easiest way to configure the embedding model is to add the  environment variables to a `.env` file in the project root:

```bash
EMBEDDING_MODEL_API_URL=https://api.example.com/v1/embeddings
EMBEDDING_MODEL_API_KEY=your_api_key_here
EMBEDDING_MODEL_NAME=jina-embeddings-v4-text-matching-GGUF
```

Embeddings are stored in the sqlite3 database after computation. Though `chromadb` is a project dependency, I am not using
ChromaDB to store the embeddings.  ChromaDB's embedding function code is used to call the embedding
and return the vector. I may refactor this dependency out later. It was very convenient!

## Topic discovery

Topic discovery is computed with the `gpt-oss-20b` model by default.
Model config is provided through environment variables, and the following parameters are needed:

- `SUMMARIZING_MODEL_API_URL` – Required: An OpenAI compatible endpoint for model access
- `SUMMARIZING_MODEL_API_KEY` - Required: The key for accessing that API
- `SUMMARIZING_MODEL_NAME` - Optional: The name of the model in the API. Defaults to `gpt-oss-20b` if not provided

## Data structure

The sqlite3 database has the following tables:

* **chunk_log** – Name, download path, and other metadata about chunks of the Wikipedia archive that can be or have been downloaded
* **page_log** – Page ID, title, abstract, and other metadata
* **page_vector** – Embedding and other computed vectors, plus cluster ID assignments for pages.
* **cluster_tree** – Cluster node info from the `recursive-cluster` command

## Visualization

Most of the work for visualization hasn't been done yet, BUT...

a test visualization of the cluster tree for English Wikipedia can be produced by running

```bash
python graph_cluster_tree.py
```

which produces `cluster_tree.html`, which you can load in the browser to view a flexible network diagram of the clusters.

## CI

This project uses [ruff](https://github.com/astral-sh/ruff) for linting (also from Astral Labs):

```bash
uvx ruff check
```

and [vulture]() to help find dead code, though the output must be evaluated by a human.

```bash
uvx vulture *.py
```