# =========================================================
# File: src/utils/mappings.py
# Description:
#   Utility functions for:
#     - TOEIC part mappings
#     - Subject category mappings
#     - Tag parsing and category assignment
#     - Metadata enrichment for recommendations
# =========================================================

import pandas as pd

# =========================================================
# 1. TOEIC Part Mapping
# =========================================================

def get_toeic_part_mapping():
    # Returns a dictionary mapping TOEIC part IDs (float) to descriptive names
    return {
        0.0: "Pre-test",
        1.0: "Part 1: Listening - Photographs",
        2.0: "Part 2: Listening - Question-Response",
        3.0: "Part 3: Listening - Conversations",
        4.0: "Part 4: Listening - Talks",
        5.0: "Part 5: Reading - Incomplete Sentences",
        6.0: "Part 6: Reading - Text Completion",
        7.0: "Part 7: Reading - Reading Comprehension"
    }

# =========================================================
# 2. Subject Category Mapping
# =========================================================

def get_subject_category_mapping():
    # Returns a dictionary mapping tag IDs to subject category names
    subject_mapping = {
        'grammar_basic': list(range(1, 23)),
        'grammar_intermediate': list(range(23, 51)),
        'grammar_advanced': [180, 181, 182, 183, 184, 185],
        'vocabulary_business': list(range(71, 105)),
        'vocabulary_general': list(range(106, 146)),
        'reading_comprehension': list(range(146, 178)),
        'listening_comprehension': list(range(187, 217)),
        'advanced_topics': list(range(217, 299))
    }
    # Reverse mapping: tag_id -> category_name
    tag_to_category = {}
    for category, tags in subject_mapping.items():
        for tag in tags:
            tag_to_category[tag] = category
    return tag_to_category

# =========================================================
# 3. Tag Parsing
# =========================================================

def parse_tags(tags_str):
    #  Parse a string of tags into a list of integers
    if pd.isna(tags_str) or tags_str == '':
        return []
    # Tags separated by commas or semicolons
    parts = [x.strip() for x in str(tags_str).replace(';', ',').split(',')]
    return [int(p) for p in parts if p.isdigit()] # List of tag IDs as integers

# =========================================================
# 4. Part & Subject Name Utilities
# =========================================================

def get_part_name(part_id):
    # Get the descriptive name for a TOEIC part ID
    part_mapping = get_toeic_part_mapping()
    return part_mapping.get(float(part_id), f"Part {part_id}")

def get_subject_categories(tags_str):
    # Convert tag string into a list of subject categories
    tag_mapping = get_subject_category_mapping()
    tags = parse_tags(tags_str)
    return list(set([tag_mapping.get(tag, "General") for tag in tags]))

# =========================================================
# 5. Lecture Title Generator
# =========================================================

def create_lecture_title(lecture_id, part, tags, video_minutes=None):
    # Get readable part name or fallback to default
    part_name = get_part_name(part) if pd.notna(part) else "General Content"
    # Extract subject categories from tags
    categories = get_subject_categories(tags)
    # Format primary subject for display
    subject = categories[0].replace('_', ' ').title() if categories else "General Topics"
    # Format duration string if valid
    duration_str = f" ({video_minutes:.2f} min)" if video_minutes is not None and pd.notna(video_minutes) and video_minutes > 0 else ""
    # Construct final lecture title
    return f"{part_name}: {subject}{duration_str}"

# =========================================================
# 6. Recommendation Metadata Enrichment
# =========================================================

def enrich_recommendations_with_metadata(recommendations, lectures_df):
    enriched_recs = []

    for rec in recommendations:
        # If already a complete dict, append as-is
        if isinstance(rec, dict) and all(key in rec for key in ['item_id', 'score', 'title', 'part', 'part_id', 'subjects', 'duration_minutes', 'type']):
            enriched_recs.append(rec)
        else:
            # If tuple/list format, unpack item_id and score
            if isinstance(rec, (list, tuple)) and len(rec) == 2:
                item_id, score = rec
                is_lecture = item_id.startswith('l')

                if is_lecture:
                    # Lookup lecture metadata from DataFrame
                    lecture_row = lectures_df[lectures_df['lecture_id'] == item_id]
                    if not lecture_row.empty:
                        row = lecture_row.iloc[0]
                        enriched_recs.append({
                            'item_id': item_id,
                            'score': round(float(score), 4),
                            'title': create_lecture_title(row['lecture_id'], row['part'], row['tags'], row.get('video_minutes')),
                            'part': get_part_name(row['part']),
                            'part_id': int(row['part']) if pd.notna(row['part']) else 0,
                            'subjects': get_subject_categories(row['tags']),
                            'duration_minutes': float(row.get('video_minutes', 0)),
                            'type': 'lecture'
                        })
                else:
                    # Fallback for bundles with minimal metadata
                    enriched_recs.append({
                        'item_id': item_id,
                        'score': round(float(score), 4),
                        'title': f"Bundle: {item_id}",
                        'part': 'General',
                        'part_id': 0,
                        'subjects': ['General'],
                        'duration_minutes': 0.0,
                        'type': 'bundle'
                    })

    return enriched_recs