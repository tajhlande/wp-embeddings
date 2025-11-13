import csv
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class LanguageDataError(Exception):
    """Custom exception for language data validation errors."""
    pass


def load_languages_from_csv(filepath: str = "languages.csv") -> Dict[str, Tuple[str, str]]:
    """
    Load language code, name, and namespace data from a CSV file.

    Args:
        filepath: Path to the CSV file containing language data

    Returns:
        Dictionary mapping language names to tuples of (iso_code, namespace)

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        LanguageDataError: If the CSV file has invalid format or data
    """
    csv_path = Path(filepath)

    if not csv_path.exists():
        raise FileNotFoundError(f"Language CSV file not found: {filepath}")

    lang_dict = {}

    try:
        with open(csv_path, encoding="utf-8") as csvfile:
            # Use csv.DictReader for better maintainability and error handling
            reader = csv.DictReader(csvfile)

            # Validate headers
            expected_headers = {'Language', 'ISO 639-1 Code', 'Namespace'}
            if not expected_headers.issubset(reader.fieldnames or set()):
                raise LanguageDataError(
                    f"Invalid CSV headers. Expected: {expected_headers}, "
                    f"Got: {set(reader.fieldnames or [])}"
                )

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    language = row['Language'].strip()
                    iso_code = row['ISO 639-1 Code'].strip()
                    namespace = row['Namespace'].strip()

                    # Validate data
                    if not all([language, iso_code, namespace]):
                        raise LanguageDataError(
                            f"Empty field in row {row_num}: {row}"
                        )

                    # Check for duplicate language names
                    if language in lang_dict:
                        logger.warning(f"Duplicate language '{language}' found in row {row_num}")

                    lang_dict[language] = (iso_code, namespace)

                except KeyError as e:
                    raise LanguageDataError(f"Missing required field in row {row_num}: {e}")

    except csv.Error as e:
        raise LanguageDataError(f"CSV parsing error: {e}")

    if not lang_dict:
        raise LanguageDataError("No language data found in CSV file")

    logger.info(f"Successfully loaded {len(lang_dict)} languages from {filepath}")
    return lang_dict


def make_namespace_to_language_index(lang_dict: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    namespace_dict = dict()
    for language in lang_dict:
        namespace_dict[lang_dict[language][1]] = language
    return namespace_dict


lang_dict: Dict[str, tuple[str, str]] = dict()
namespace_dict: Dict[str, str] = dict()


def get_language_for_namespace(namespace: str) -> str:
    """
    Given a namespace, get the name of the corresponding language for it.
    Loads the language data from a CSV file, and uses global dict variables to cache it.
    """
    global namespace_dict
    if not namespace_dict:
        global lang_dict
        if not lang_dict:
            lang_dict = load_languages_from_csv()
        namespace_dict = make_namespace_to_language_index(lang_dict)
    return namespace_dict[namespace]
