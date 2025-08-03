import pytest
import pandas as pd
from src.utils.mappings import (
    get_part_name,
    get_subject_categories,
    create_lecture_title,
    enrich_recommendations_with_metadata
)

# --- get_part_name ---
def test_get_part_name_known():
    assert get_part_name(1) == "Part 1: Listening - Photographs"

def test_get_part_name_unknown():
    assert get_part_name(99) == "Part 99"

# --- get_subject_categories ---
def test_get_subject_categories_basic():
    tag_str = "3,5,8"
    result = get_subject_categories(tag_str)
    assert isinstance(result, list)
    assert "grammar_basic" in result

def test_get_subject_categories_empty():
    assert get_subject_categories("") == []

# --- create_lecture_title ---
def test_create_lecture_title_basic():
    title = create_lecture_title("l123", 1, "3,5", 7.5)
    assert isinstance(title, str)
    assert "Listening" in title
    assert "Grammar" in title
    assert "(7.50 min)" in title

# --- enrich_recommendations_with_metadata ---
def test_enrich_recommendations_basic():
    lectures_df = pd.DataFrame([{
        "lecture_id": "l100",
        "part": 1,
        "tags": "3,5",
        "video_minutes": 5.0
    }])
    raw_recs = [("l100", 0.95)]
    enriched = enrich_recommendations_with_metadata(raw_recs, lectures_df)
    assert len(enriched) == 1
    assert enriched[0]["item_id"] == "l100"
    assert "title" in enriched[0]
    assert enriched[0]["type"] == "lecture"