import logging

from classes import ClusterNodeTopics, Page, PageContent
from database import (
    get_cluster_node_first_pass_topic,
    get_cluster_parent_id,
    get_neighboring_first_topics,
    get_pages_in_cluster,
    get_sql_conn,
    _row_to_dataclass
)
from topic_discovery import TopicDiscovery, get_system_prompt_for_namespace


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    namespace = "enwiki_namespace_0"
    sqlconn = get_sql_conn(namespace)
    cursor = sqlconn.cursor()
    cursor.execute(page_list_sql)
    rows = cursor.fetchall()
    pages = [_row_to_dataclass(row, Page) for row in rows]

    logger.info("Summarizing topics for %d pages", len(pages))
    cnt = ClusterNodeTopics(node_id=3, depth=3, parent_id=2,
                            first_label="first", final_label="final", is_leaf=True)
    page_topics = [PageContent(page_id=page.page_id, title=page.title, abstract=page.abstract or "") for page in pages]
    topic = topic_discovery.naively_summarize_page_topics(namespace, node=cnt, pages=page_topics)

    logger.info("Summarized topic: %s", topic)


def test_adverse_topic_summary():
    """
    Some leaf cluster IDs:
        3
        5
        7
        8
        11
        12
        13
        15
        16
        17
    """

    namespace = "enwiki_namespace_0"
    sqlconn = get_sql_conn(namespace)
    cluster_id = 3
    parent_id = get_cluster_parent_id(sqlconn, namespace, cluster_id)
    assert parent_id is not None
    logger.debug(f"finding adverse topic for cluster {cluster_id} with parent {parent_id}")
    first_topic = get_cluster_node_first_pass_topic(sqlconn, namespace, cluster_id)
    logger.debug(f"First pass topic: {first_topic}")
    page_list = get_pages_in_cluster(sqlconn, namespace, cluster_id)

    logger.debug(f"Page titles: {', '.join([page.title for page in page_list[:5]])}, ...")

    neighboring_topic_list = get_neighboring_first_topics(sqlconn, namespace, cluster_id, parent_id)
    logger.debug(f"Neighboring topics: {', '.join(neighboring_topic_list[:5])}, ...")

    topic_discovery = TopicDiscovery.get_from_env()

    cnt = ClusterNodeTopics(node_id=3, depth=3, parent_id=2,
                            first_label="first", final_label="final", is_leaf=True)
    page_topics = [PageContent(page_id=page.page_id, title=page.title, abstract=page.abstract or "") for page in page_list]
    n_id = 5
    neighbor_nodes = [ClusterNodeTopics(node_id=(n_id := n_id + 1), depth=3, parent_id=2,
                                        first_label="first", final_label="final", is_leaf=True
                                        ) for topic in neighboring_topic_list]
    final_topic = topic_discovery.adversely_summarize_page_topics(namespace, node=cnt, pages=page_topics,
                                                                  siblings=neighbor_nodes)
    logger.debug(f"Final pass topic: {final_topic}")


def test_system_prompt_generation():
    generated_prompt = get_system_prompt_for_namespace("dewiki_namespace_0")
    assert "Generated output should only be in German." in generated_prompt

    generated_prompt = get_system_prompt_for_namespace("enwiki_namespace_0")
    assert "Generated output should only be in English." in generated_prompt

    generated_prompt = get_system_prompt_for_namespace("arzwiki_namespace_0")
    assert "Generated output should only be in Egyptian Arabic." in generated_prompt


# if __name__ == "__main__":
#     test_adverse_topic_summary()
