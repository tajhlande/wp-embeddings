import logging
import os
from typing import Optional
from dotenv import load_dotenv

from openai import OpenAI

from classes import Page

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOPIC_GENERATION_SYSTEM_PROMPT = " ".join("""
    You are a multi-lingual expert summarization assistant.
    For any collection of submitted page titles, cluster topics, and abstract information, you can
    succinctly produce a topic description of the content.
    You should be using the same language as the most common language in the
    submitted titles, topics, and abstracts.
    The generated topic must be the best description of the subject matter area across all the given
    materials, while maintaining clarity and distinctiveness from other topics.
""".split())

SUMMARIZING_MODEL_NAME_KEY = "SUMMARIZING_MODEL_NAME"
SUMMARIZING_MODEL_API_URL_KEY = "SUMMARIZING_MODEL_API_URL"
SUMMARIZING_MODEL_API_KEY_KEY = "SUMMARIZING_MODEL_API_KEY"
DEFAULT_MODEL_NAME = "gpt-oss-20b"


class TopicDiscovery:

    _openai_client: OpenAI
    model_name: str

    def __init__(
        self,
        ai_server_base_url: str,
        ai_server_key: Optional[str],
        model_name: str = DEFAULT_MODEL_NAME
    ):
        logger.info("Opening connection to OpenAI compatible server at %s", ai_server_base_url)
        self._openai_client = OpenAI(
           api_key=ai_server_key,
           base_url=ai_server_base_url
        )
        self.model_name = model_name

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

    def summarize_page_topics(self, page_list: list[Page]) -> Optional[str]:
        # TODO let's just use page titles for now. we will introduce page abstract content later if we need it
        submitted_titles = ", ".join([page.title for page in page_list])

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
            The page titles are: {submitted_titles}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        completion = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": TOPIC_GENERATION_SYSTEM_PROMPT, },
                {"role": "user", "content": prompt, },
            ],
        )

        return completion.choices[0].message.content

        # response = self._openai_client.responses.create(
        #     model=self.model_name,
        #     instructions=TOPIC_GENERATION_SYSTEM_PROMPT,
        #     input=prompt,
        # )

        # return response.output_text

    def adversely_summarize_page_topics(self,
                                        page_list: list[Page],
                                        neighboring_topics_list: list[str]
                                        ) -> Optional[str]:
        submitted_titles = ", ".join([page.title for page in page_list])
        neighboring_topics = ", ".join(neighboring_topics_list)

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
            The page titles are: {submitted_titles}.
            The neighboring topics from which the generated topic should be distinct are: {neighboring_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        completion = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": TOPIC_GENERATION_SYSTEM_PROMPT, },
                {"role": "user", "content": prompt, },
            ],
        )

        return completion.choices[0].message.content

    def summarize_cluster_topics(self, child_topics: list[str]) -> Optional[str]:
        submitted_topics = ", ".join([topic for topic in child_topics if topic is not None])

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
            The cluster topic are: {submitted_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        completion = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": TOPIC_GENERATION_SYSTEM_PROMPT, },
                {"role": "user", "content": prompt, },
            ],
        )

        return completion.choices[0].message.content

    def adversely_summarize_cluster_topics(self,
                                           child_topics: list[str],
                                           neighboring_topics_list: list[str]
                                           ) -> Optional[str]:
        submitted_titles = ", ".join([topic for topic in child_topics if topic is not None])
        neighboring_topics = ", ".join(neighboring_topics_list)

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
            The page titles are: {submitted_titles}.
            The neighboring topics from which the generated topic should be distinct are: {neighboring_topics}.
        """.split())

        logger.debug("Prompt: %s", prompt)

        completion = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": TOPIC_GENERATION_SYSTEM_PROMPT, },
                {"role": "user", "content": prompt, },
            ],
        )

        return completion.choices[0].message.content
