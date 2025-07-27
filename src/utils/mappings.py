# src/utils/mappings.py

import pandas as pd
import numpy as np

def get_toeic_part_mapping():
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

def get_subject_category_mapping():
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
    tag_to_category = {}
    for category, tags in subject_mapping.items():
        for tag in tags:
            tag_to_category[tag] = category
    return tag_to_category

def parse_tags(tags_str):
    if pd.isna(tags_str) or tags_str == '':
        return []
    parts = [x.strip() for x in str(tags_str).replace(';', ',').split(',')]
    return [int(p) for p in parts if p.isdigit()]

def get_part_name(part_id):
    part_mapping = get_toeic_part_mapping()
    return part_mapping.get(float(part_id), f"Part {part_id}")

def get_subject_categories(tags_str):
    tag_mapping = get_subject_category_mapping()
    tags = parse_tags(tags_str)
    return list(set([tag_mapping.get(tag, "General") for tag in tags]))

def create_lecture_title(lecture_id, part, tags, video_minutes=None):
    part_name = get_part_name(part) if pd.notna(part) else "General Content"
    categories = get_subject_categories(tags)
    subject = categories[0].replace('_', ' ').title() if categories else "General Topics"
    duration_str = f" ({video_minutes:.2f} min)" if video_minutes is not None and pd.notna(video_minutes) and video_minutes > 0 else ""
    return f"{part_name}: {subject}{duration_str}"

def enrich_recommendations_with_metadata(recommendations, lectures_df):
    enriched_recs = []
    for rec in recommendations:
        if isinstance(rec, dict) and all(key in rec for key in ['item_id', 'score', 'title', 'part', 'part_id', 'subjects', 'duration_minutes', 'type']):
            enriched_recs.append(rec)
        else:
            if isinstance(rec, (list, tuple)) and len(rec) == 2:
                item_id, score = rec
                is_lecture = item_id.startswith('l')
                if is_lecture:
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
                    # Assume bundle; minimal metadata
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