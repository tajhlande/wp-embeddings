import logging

from classes import Page
from database import get_sql_conn, _row_to_dataclass
from topic_discovery import TopicDiscovery


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_topic_summary_llm_call():
    topic_discovery = TopicDiscovery.get_from_env()

    page_list_sql = """
        SELECT page_log.namespace, page_log.page_id, title, chunk_name, url, extracted_at, abstract
        FROM page_log
        INNER JOIN page_vector ON page_log.namespace = page_vector.namespace
        AND page_log.page_id = page_vector.page_id
        WHERE page_log.namespace = 'enwiki_namespace_0'
        AND page_log.page_id > 200000
        ORDER BY page_log.page_id ASC
        LIMIT 200
    """
    #        AND page_vector.cluster_node_id = 10

    sqlconn = get_sql_conn()
    cursor = sqlconn.cursor()
    cursor.execute(page_list_sql)
    rows = cursor.fetchall()
    pages = [_row_to_dataclass(row, Page) for row in rows]

    logger.info("Summarizing topics for %d pages", len(pages))
    topic = topic_discovery.summarize_page_topics(page_list=pages)

    logger.info("Summarized topic: %s", topic)


if __name__ == "__main__":
    test_topic_summary_llm_call()
