import numpy as np
import pandas as pd


def tag_mapping(taxonomy):
    tag_mapping = {}

    for _, row in taxonomy.iterrows():
        first_level = row['Уровень 1 (iab)']
        second_level = row['Уровень 2 (iab)']
        if first_level not in tag_mapping:
            tag_mapping[first_level] = set()
        if pd.notna(second_level):
            tag_mapping[first_level].add(second_level)
    return tag_mapping


def extract_tags(tags):
    first_levels = set()
    second_levels = set()
    for tag in tags.split(','):
        parts = tag.split(':')
        first_levels.add(parts[0].strip())
        if len(parts) > 1:
            second_levels.add(parts[1].strip())
    return list(first_levels), list(second_levels)

# Фильтрация второго уровня по первому
def filter_second_level_by_first_level(preds_second_level, preds_first_level, mlb_first_level, mlb_second_level, mapping):
    filtered_second_level_tags = []
    for i in range(len(preds_second_level)):
        predicted_first_tags = mlb_first_level.classes_[preds_first_level[i].astype(bool)]
        allowed_second_level_tags = set()

        for tag in predicted_first_tags:
            if tag in mapping:
                allowed_second_level_tags.update(mapping[tag])

        allowed_indices = mlb_second_level.transform([list(allowed_second_level_tags)])[0].astype(bool)
        filtered_pred = preds_second_level[i] * allowed_indices
        filtered_second_level_tags.append(filtered_pred)

    return np.array(filtered_second_level_tags)
