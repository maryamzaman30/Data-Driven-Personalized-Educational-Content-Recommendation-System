# =========================================================
# File: tests/test_mappings.py
# Description:
#   Unit tests for src.utils.mappings utility functions.
#   Covers:
#       - get_part_name
#       - get_subject_categories
#       - create_lecture_title
#       - enrich_recommendations_with_metadata
# =========================================================

import pandas as pd
from src.utils.mappings import (
    get_part_name,
    get_subject_categories,
    create_lecture_title,
    enrich_recommendations_with_metadata
)

# =========================================================
# 1. Tests for get_part_name
# =========================================================

def test_get_part_name_known():
    """
    Should return a descriptive part name for a known part number.
    """
    assert get_part_name(1) == "Part 1: Listening - Photographs"


def test_get_part_name_unknown():
    """
    Should return 'Part X' format when part number is unknown.
    """
    assert get_part_name(99) == "Part 99"

# =========================================================
# 2. Tests for get_subject_categories
# =========================================================

def test_get_subject_categories_basic():
    """
    Should return a list of subject categories based on tag string.
    """
    tag_str = "3,5,8"
    result = get_subject_categories(tag_str)

    # Check result type
    assert isinstance(result, list)

    # Should contain expected category
    assert "grammar_basic" in result


def test_get_subject_categories_empty():
    """
    Should return an empty list when tag string is empty.
    """
    assert get_subject_categories("") == []


# =========================================================
# 3. Tests for create_lecture_title
# =========================================================

def test_create_lecture_title_basic():
    """
    Should generate a readable lecture title with:
        - Part name
        - Subject categories
        - Video duration
    """
    title = create_lecture_title("l123", 1, "3,5", 7.5)

    # Validate type
    assert isinstance(title, str)

    # Validate expected content
    assert "Listening" in title
    assert "Grammar" in title
    assert "(7.50 min)" in title

# =========================================================
# 4. Tests for enrich_recommendations_with_metadata
# =========================================================

def test_enrich_recommendations_basic():
    """
    Should enrich raw recommendations with lecture metadata.
    """
    # Prepare mock lectures data
    lectures_df = pd.DataFrame([{
        "lecture_id": "l100",
        "part": 1,
        "tags": "3,5",
        "video_minutes": 5.0
    }])

    # Raw recommendations: (item_id, score)
    raw_recs = [("l100", 0.95)]

    # Enrich recommendations
    enriched = enrich_recommendations_with_metadata(raw_recs, lectures_df)

    # Validate length
    assert len(enriched) == 1

    # Validate enriched structure
    assert enriched[0]["item_id"] == "l100"
    assert "title" in enriched[0]
    assert enriched[0]["type"] == "lecture"