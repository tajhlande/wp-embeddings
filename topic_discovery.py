import logging
import os
import concurrent.futures

from random import random
import time
from typing import Dict, Iterable, Optional
from dotenv import load_dotenv

from openai import OpenAI, InternalServerError

from classes import ClusterNodeTopics, PageContent
from languages import get_language_for_namespace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpcore_logger = logging.getLogger("httpcore")
httpcore_logger.setLevel(logging.WARNING)

TOPIC_GENERATION_SYSTEM_PROMPT = " ".join("""
    You are a multi-lingual expert summarization assistant.
    For any collection of submitted page titles, cluster topics, and abstract information, you can
    succinctly produce a topic description of the content.
    The generated topic must be the best description of the subject matter area across all the given
    materials, while maintaining clarity and distinctiveness from other topics.
""".split())
# moving this part out because we'll describe the language per the namespace.
"""    You should be using the same language as the most common language in the
    submitted titles, topics, and abstracts."""

SUMMARIZING_MODEL_NAME_KEY = "SUMMARIZING_MODEL_NAME"
SUMMARIZING_MODEL_API_URL_KEY = "SUMMARIZING_MODEL_API_URL"
SUMMARIZING_MODEL_API_KEY_KEY = "SUMMARIZING_MODEL_API_KEY"
DEFAULT_MODEL_NAME = "gpt-oss-20b"

APP_TITLE = "wp-embeddings"
# APP_URL = "http://localhost:8080"

namespace_to_system_prompt_dict: Dict[str, str] = dict()


def get_system_prompt_for_namespace(namespace: str) -> str:
    if namespace in namespace_to_system_prompt_dict:
        return namespace_to_system_prompt_dict[namespace]

    language = get_language_for_namespace(namespace)

    prompt = f"{TOPIC_GENERATION_SYSTEM_PROMPT} " \
             f"Generated output should only be in {language}."
    namespace_to_system_prompt_dict[namespace] = prompt
    return prompt


class TopicDiscovery:

    _openai_client: OpenAI
    model_name: str
    accumulated_errors: int = 0
    _max_workers: int = 8

    def __init__(
        self,
        ai_server_base_url: str,
        ai_server_key: Optional[str],
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        logger.info("Opening connection to OpenAI compatible server at %s", ai_server_base_url)
        self._openai_client = OpenAI(
           api_key=ai_server_key,
           base_url=ai_server_base_url
        )
        self.model_name = model_name
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)

    @classmethod
    def get_from_env(cls):
        load_dotenv()
        model_name = os.environ.get(SUMMARIZING_MODEL_NAME_KEY, DEFAULT_MODEL_NAME)
        api_url = os.environ.get(SUMMARIZING_MODEL_API_URL_KEY)
        api_key = os.environ.get(SUMMARIZING_MODEL_API_KEY_KEY)

        if not api_url:
            raise ValueError(f"No value for {SUMMARIZING_MODEL_API_URL_KEY} found in the environment")

        return cls(
            ai_server_base_url=api_url,
            ai_server_key=api_key,
            model_name=model_name
            )

    def _call_openai_completions_endpoint(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call the openai client and invoke the completions endpoint.
        Do a few retries with truncated user prompt if 500-599 errors are caught
        """
        attempts = 0
        max_attempts = 5
        previous_error = None
        original_user_prompt = user_prompt
        while attempts < max_attempts:
            try:
                attempts += 1
                completion = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "developer", "content": system_prompt, },
                        {"role": "user", "content": user_prompt, },
                    ],
                    extra_body={
                        # llama.cpp-specific slot parameters to deal with "500: context shift is disabled" errors
                        "cache_prompt": False,
                        "n_keep": 0,
                        # OpenRouter.ai specific parameters
                        "provider": {
                            # "order": ["novita/bf16", "ncompass", "gmicloud/fp4", "deepinfra/fp4", ]  # gpt-oss-120b
                            "order": ["novita", "deepinfra", "wandb", ],  # gpt-oss-20b preferences
                        },
                    },
                    extra_headers={
                        # Headers for OpenRouter.ai
                        # "HTTP-Referer": APP_URL,
                        "X-Title": APP_TITLE,
                    },
                    temperature=1.0,
                    top_p=1.0,
                )
                return completion.choices[0].message.content

            except InternalServerError as e:
                self.accumulated_errors += 1
                previous_error = e
                if e.status_code >= 500 and e.status_code <= 599:
                    # random sleep fallback of 25-225 ms
                    time.sleep((random() * 200 + 25)/1000)
                else:
                    # we only retry internal server errors
                    logger.error("Giving up on handling openai.InternalServerError after %d attempts", attempts)
                    logger.error("System prompt (%d chars): %s", len(system_prompt), system_prompt)
                    logger.error("Last user prompt (%d chars): %s", len(user_prompt), user_prompt)
                    logger.error("Original user prompt (%d chars): %s", len(original_user_prompt), original_user_prompt)
                    raise
                # if the user prompt was long, cut it in half for the retry
                if len(user_prompt) > 250:
                    user_prompt = user_prompt[:len(user_prompt) // 2]

        if previous_error:
            # raise the unsolvable error
            raise previous_error

        logger.warning("OpenAI server call failed multiple times but no previous error present")
        return None

    def naively_summarize_page_topics(self, namespace: str, node: ClusterNodeTopics, pages: list[PageContent]) -> None:
        # Using page title and abstract clip for summaries
        # only first 100 pages. leaf nodes shouldn't be larger than that anyway
        submitted_titles = "; ".join([page.title + (f": {page.abstract}" if page.abstract else "")
                                      for page in pages[:100]])

        prompt = " ".join(f"""
            Describe the best common topic for the page titles listed below.
            Only generate a topic, and no other content.
            Use a few words or word-equivalents if the language is non-Latin.
            Be descriptive, succinct but not terse.
            Do not generate any other text besides the topic.
            Only use punctuation as appropriate for the words in the topic.
            You do not need to produce a complete sentence and should not end the topic with a period
            or other sentence terminator.
            The first word in the topic should be capitalized, if the language has capitalization.
            The page titles and abstracts are: {submitted_titles}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        node.first_label = self._call_openai_completions_endpoint(get_system_prompt_for_namespace(namespace), prompt)
        node.first_label = node.first_label.strip() if node.first_label else None

    def adversely_summarize_page_topics(self,
                                        namespace: str,
                                        node: ClusterNodeTopics,
                                        pages: list[PageContent],
                                        siblings: list[ClusterNodeTopics]
                                        ):
        # use title and abstract
        submitted_titles = "; ".join([page.title + (f": {page.abstract}" if page.abstract else "")
                                      for page in pages[:100]])
        neighboring_topics = ", ".join([sib.first_label for sib in siblings if sib.first_label])

        prompt = " ".join(f"""
            Describe the best common topic for the following page titles.
            Consider also the neighboring topics, and generate a topic that
            is as distinct from the neighboring topics as it can reasonably be, without distorting
            the topic's descriptive power.
            Only generate a topic, and no other content.
            Use a few words or word-equivalents if the language is non-Latin.
            Be descriptive, succinct but not terse.
            Do not generate any other text besides the topic.
            Only use punctuation as appropriate for the words in the topic.
            You do not need to produce a complete sentence and should not end the topic with a period
            or other sentence terminator.
            The first word in the topic should be capitalized, if the language has capitalization.
            The page titles are: {submitted_titles}.
            The neighboring topics from which the generated topic should be distinct are: {neighboring_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        node.final_label = self._call_openai_completions_endpoint(get_system_prompt_for_namespace(namespace), prompt)
        node.final_label = node.final_label.strip() if node.final_label else None

    def naively_summarize_cluster_topics(self, namespace: str,
                                         node: ClusterNodeTopics,
                                         child_nodes: list[ClusterNodeTopics]):
        submitted_topics = ", ".join([c.first_label for c in child_nodes if c.first_label])

        prompt = " ".join(f"""
            Describe the best common topic for the cluster topics listed below.
            Only generate a topic, and no other content.
            Use a few words or word-equivalents if the language is non-Latin.
            Be descriptive, succinct but not terse.
            Do not generate any other text besides the topic.
            Only use punctuation as appropriate for the words in the topic.
            You do not need to produce a complete sentence and should not end the topic with a period
            or other sentence terminator.
            The first word in the topic should be capitalized, if the language has capitalization.
            The cluster topics are: {submitted_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        node.first_label = self._call_openai_completions_endpoint(get_system_prompt_for_namespace(namespace), prompt)
        node.first_label = node.first_label.strip() if node.first_label else None

    def adversely_summarize_cluster_topics(self,
                                           namespace: str,
                                           node: ClusterNodeTopics,
                                           child_nodes: list[ClusterNodeTopics],
                                           sibling_nodes: list[ClusterNodeTopics]
                                           ):
        cluster_topics = ", ".join([c.final_label for c in child_nodes if c.final_label])
        neighboring_topics = ", ".join([s.first_label for s in sibling_nodes if s.first_label])

        prompt = " ".join(f"""
            Describe the best common topic for the following cluster topics.
            Consider also the neighboring topics, and generate a topic that
            is as distinct from the neighboring topics as it can reasonably be, without distorting
            the topic's descriptive power.
            Only generate a topic, and no other content.
            Use a few words or word-equivalents if the language is non-Latin.
            Be descriptive, succinct but not terse.
            Do not generate any other text besides the topic.
            Only use punctuation as appropriate for the words in the topic.
            You do not need to produce a complete sentence and should not end the topic with a period
            or other sentence terminator.
            The first word in the topic should be capitalized, if the language has capitalization.
            The cluster topics are: {cluster_topics}.
            The neighboring topics from which the generated topic should be distinct are: {neighboring_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        node.final_label = self._call_openai_completions_endpoint(get_system_prompt_for_namespace(namespace), prompt)
        node.final_label = node.final_label.strip() if node.final_label else None

    # -----------------------------------------------------------------------
    # new batch interface (used by TopicsCommand)
    # -----------------------------------------------------------------------

    def naively_summarize_page_topics_batch(self,
                                            namespace: str,
                                            batch: Iterable[ClusterNodeTopics],
                                            page_content_map: dict[int, list[PageContent]]
                                            ):
        futures = [
            self._executor.submit(self.naively_summarize_page_topics, namespace, node, page_content_map[node.node_id])
            for node in batch
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]

    def _get_siblings(self,
                      node: ClusterNodeTopics,
                      parent_children: list[ClusterNodeTopics]
                      ) -> list[ClusterNodeTopics]:
        return [
            s for s in parent_children
            if s.node_id != node.node_id
        ]

    def adversely_summarize_page_topics_batch(self,
                                              namespace: str,
                                              batch: Iterable[ClusterNodeTopics],
                                              page_content_map: dict[int, list[PageContent]],
                                              parent_id_child_map: dict[int, list[ClusterNodeTopics]]
                                              ):
        futures = [
            self._executor.submit(self.adversely_summarize_page_topics,
                                  namespace, node, page_content_map[node.node_id],
                                  self._get_siblings(node, parent_id_child_map[node.parent_id]))
            for node in batch if node.parent_id
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]

    def naively_summarize_cluster_topics_batch(self,
                                               namespace: str,
                                               batch: Iterable[ClusterNodeTopics],
                                               parent_id_child_map: dict[int, list[ClusterNodeTopics]]):
        futures = [
            self._executor.submit(self.naively_summarize_cluster_topics,
                                  namespace=namespace,
                                  node=node,
                                  child_nodes=parent_id_child_map[node.node_id])
            for node in batch
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]

    def adversely_summarize_cluster_topics_batch(self,
                                                 namespace: str,
                                                 batch: list[ClusterNodeTopics],
                                                 parent_id_child_map: dict[int, list[ClusterNodeTopics]]
                                                 ) -> list[Optional[str]]:
        """
        Given a batch of tuples, do adverse summarization.
        The first tuple entry is a list of child topics for a cluster node.
        The second tuple entry is a list of neighbring topics for a cluster node.
        """
        futures = [
            self._executor.submit(self.adversely_summarize_cluster_topics,
                                  namespace, node, parent_id_child_map[node.node_id],
                                  self._get_siblings(node, parent_id_child_map[node.parent_id]))
            for node in batch if node.parent_id
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]
