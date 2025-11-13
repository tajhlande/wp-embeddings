import pytest
from languages import (
    load_languages_from_csv,
    LanguageDataError,
    make_namespace_to_language_index,
    get_language_for_namespace
)


class TestLanguageMaps:
    """Test suite for language loading functionality."""

    @pytest.fixture
    def expected_languages(self):
        """Define expected test data for validation."""
        return {
            "Chinese": ("zh", "zhwiki_namespace_0"),
            "English": ("en", "enwiki_namespace_0"),
            "French": ("fr", "frwiki_namespace_0"),
            "Spanish": ("es", "eswiki_namespace_0"),
            "Portuguese": ("pt", "ptwiki_namespace_0"),
            "German": ("de", "dewiki_namespace_0"),
            "Italian": ("it", "itwiki_namespace_0"),
            "Russian": ("ru", "ruwiki_namespace_0"),
            "Japanese": ("ja", "jawiki_namespace_0"),
            "Korean": ("ko", "kowiki_namespace_0"),
            "Vietnamese": ("vi", "viwiki_namespace_0"),
            "Thai": ("th", "thwiki_namespace_0"),
            "Arabic": ("ar", "arwiki_namespace_0"),
            "Egyptian Arabic": ("arz", "arzwiki_namespace_0")
        }

    def test_load_languages_success(self, expected_languages):
        """Test successful loading of language data."""
        lang_dict = load_languages_from_csv()

        # Test count
        assert len(lang_dict) == len(expected_languages), \
            f"Expected {len(expected_languages)} languages, got {len(lang_dict)}"

        # Test specific language data
        for lang_name, expected_data in expected_languages.items():
            assert lang_name in lang_dict, f"Missing language: {lang_name}"
            assert lang_dict[lang_name] == expected_data, \
                f"Data mismatch for {lang_name}: expected {expected_data}, got {lang_dict[lang_name]}"

    def test_file_not_found_error(self):
        """Test handling of missing CSV file."""
        with pytest.raises(FileNotFoundError):
            load_languages_from_csv("nonexistent_file.csv")

    def test_invalid_csv_format(self, tmp_path):
        """Test handling of invalid CSV format."""
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("Invalid,CSV,Format\nToo,Few,Columns\n")

        with pytest.raises(LanguageDataError):
            load_languages_from_csv(str(invalid_csv))

    def test_missing_headers(self, tmp_path):
        """Test handling of CSV with missing headers."""
        missing_headers_csv = tmp_path / "missing_headers.csv"
        missing_headers_csv.write_text("Chinese,zh,zhwiki_namespace_0\n")

        with pytest.raises(LanguageDataError):
            load_languages_from_csv(str(missing_headers_csv))

    def test_empty_fields(self, tmp_path):
        """Test handling of empty fields in CSV."""
        empty_fields_csv = tmp_path / "empty_fields.csv"
        empty_fields_csv.write_text(
            "Language,ISO 639-1 Code,Namespace\nChinese,zh,\nEnglish,,enwiki_namespace_0\n"
        )

        with pytest.raises(LanguageDataError):
            load_languages_from_csv(str(empty_fields_csv))

    def test_namespace_map(self):
        lang_dict = load_languages_from_csv()
        namespace_dict = make_namespace_to_language_index(lang_dict)
        assert namespace_dict["enwiki_namespace_0"] == "English"
        assert namespace_dict["dewiki_namespace_0"] == "German"
        assert namespace_dict["arzwiki_namespace_0"] == "Egyptian Arabic"

    def test_language_lookup(self):
        assert get_language_for_namespace("enwiki_namespace_0") == "English"
        assert get_language_for_namespace("frwiki_namespace_0") == "French"
        assert get_language_for_namespace("arzwiki_namespace_0") == "Egyptian Arabic"
        assert get_language_for_namespace("dewiki_namespace_0") == "German"
