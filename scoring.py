BASELINE = 45

import pandas as pd
import numpy as np
from pathlib import Path

from ingredients_loader import (
    df_ingredients,
    correct_ingredients,
    filter_non_base_ingredients,
    get_ingredient_guides
)

from concern_classifier_llm import df_concerns, label_col, text_col, ensemble_predict
from skin_type_loader import df_types

# ---------------------------------------------------
# 1) 고민 → 효능 매핑용 엑셀 로딩
# ---------------------------------------------------
SKIN_CONCERN_FILE = Path("data/skin_concerns_canon.xlsx")
df_concern_map = pd.read_excel(SKIN_CONCERN_FILE)

def build_concern_map(df):
    df = df[['피부고민', '효능']].dropna()
    df['피부고민'] = df['피부고민'].astype(str).str.strip()
    df['효능'] = df['효능'].astype(str)

    tmp = {}
    for _, row in df.iterrows():
        label = row['피부고민']
        effects = [e.strip() for e in row['효능'].split(",") if e.strip()]
        if label not in tmp:
            tmp[label] = set()
        tmp[label].update(effects)

    return {k: sorted(list(v)) for k, v in tmp.items()}

CONCERN_TO_EFFECTS = build_concern_map(df_concern_map)

def get_concern_effects(label):
    return CONCERN_TO_EFFECTS.get(label, [])


# ---------------------------------------------------
# 2) 피부타입 → 효능 리스트
# ---------------------------------------------------
def get_skin_type_effects(type_name: str):
    row = df_types[df_types['피부타입'] == type_name].iloc[0]
    effects = row.drop("피부타입").dropna().tolist()
    return effects


# ---------------------------------------------------
# 3) 공통 점수 계산
# ---------------------------------------------------
def calc_match_score(ingredients, effect_list):
    if effect_list is None:
        effect_list = []
    elif isinstance(effect_list, str):
        effect_list = [e.strip() for e in effect_list.split(",") if e.strip()]
    else:
        effect_list = [str(e).strip() for e in list(effect_list) if str(e).strip()]

    effect_set = set(effect_list)

    matched = []
    scores = []

    for ing in ingredients:
        rows = df_ingredients[
            (df_ingredients['성분명'] == ing) &
            (df_ingredients['종류'] != '보조 성분')
        ]

        if len(rows) == 0:
            matched.append({
                "성분": ing,
                "성분효능": [],
                "일치효능": [],
                "전체효과개수": 0,
                "일치도": 0.0
            })
            scores.append(0.0)
            continue

        raw_effects = rows['효과별'].dropna().unique().tolist()
        ing_effects = []
        for v in raw_effects:
            parts = [e.strip() for e in str(v).split(",") if e.strip()]
            ing_effects.extend(parts)

        ing_effects = list(set(ing_effects))
        total = len(ing_effects)

        if total == 0:
            matched.append({
                "성분": ing,
                "성분효능": [],
                "일치효능": [],
                "전체효과개수": 0,
                "일치도": 0.0
            })
            scores.append(0.0)
            continue

        intersection = list(set(ing_effects) & effect_set)
        score = len(intersection) / total

        matched.append({
            "성분": ing,
            "성분효능": ing_effects,
            "일치효능": intersection,
            "전체효과개수": total,
            "일치도": round(score, 3)
        })
        scores.append(score)

    df = pd.DataFrame(matched)
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return mean_score, df


# ---------------------------------------------------
# 4) full_pipeline (웹과 완전 호환)
# ---------------------------------------------------
def full_pipeline(review, ingredients, skin_num):
    # 0) 전성분 문자열 통일
    if isinstance(ingredients, list):
        ingredients_text = ",".join(ingredients)
    else:
        ingredients_text = ingredients

    # 1) 고민 예측
    primary, probs = ensemble_predict(review)

    # 2) 성분 교정
    corrected = correct_ingredients(ingredients_text)
    if corrected is None:
        corrected = []

    # 3) 성분 가이드라인 추출
    ingredient_guides = get_ingredient_guides(corrected)

    # 4) 고민 → 효능 매핑
    concern_effects = get_concern_effects(primary)

    # 5) 피부타입 번호 → 이름
    skin_map = {1: "지성", 2: "복합성", 3: "건성", 4: "민감성"}
    stype = skin_map[skin_num]

    # 6) 고민/피부타입 점수 계산
    concern_score, concern_df = calc_match_score(corrected, concern_effects)

    # 피부타입 점수: 보조 성분 제외
    type_score, type_df = calc_match_score(
        filter_non_base_ingredients(corrected),
        get_skin_type_effects(stype)
    )

    # 7) 최종 점수
    raw_score = (concern_score * 0.65 + type_score * 0.35) * 100
    final = BASELINE + raw_score
    
    # ✅ 100점 넘으면 100으로 자르기 (원하면 0점 이하도 방어)
    if final > 100:
        final = 100
    if final < 0:   # 이 라인은 선택이지만 안전하게 두는 걸 추천
        final = 0

    # 8) 반환 (기존 FastAPI 응답 구조 유지)
    return {
        "피부타입": stype,
        "예측고민": primary,
        "고민확률": probs,
        "최종점수": round(final, 2),
        "성분가이드": ingredient_guides}