import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 1) SBERT 모델 & DB 로딩
# -----------------------------
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

df_ingredients = pd.read_excel("data/final_ingredients_dataset_canon_filled_2.xlsx")

# 성분명 전처리
df_ingredients['성분명'] = (
    df_ingredients['성분명']
    .astype(str)
    .str.replace(r"[^가-힣A-Za-z0-9\-/]", "", regex=True)
)

ingredient_names = df_ingredients['성분명'].tolist()
ingredient_embs = model.encode(ingredient_names, convert_to_tensor=True)


# -----------------------------
# 2) 전성분 문자열 정제 + split
# -----------------------------
def clean_and_split(text):
    text = re.sub(r"[^가-힣A-Za-z0-9\-/(), ]", " ", text)
    parts = re.split(r",", text)

    cleaned = [
        re.sub(r"[^가-힣A-Za-z0-9\-/]", "", p).strip()
        for p in parts
    ]
    return [c for c in cleaned if len(c) >= 2]


# -----------------------------
# 3) SBERT 성분 교정
# -----------------------------
def correct_ingredients(text, threshold=0.68):
    cand = clean_and_split(text)

    if len(cand) == 0:
        return []

    emb = model.encode(cand, convert_to_tensor=True)
    sims = util.cos_sim(emb, ingredient_embs)

    corrected = []
    for i, w in enumerate(cand):
        best_idx = torch.argmax(sims[i]).item()
        best_score = sims[i][best_idx].item()
        if best_score >= threshold:
            corrected.append(ingredient_names[best_idx])

    # 순서 유지하며 중복 제거
    seen = set()
    unique = []
    for x in corrected:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    return unique


# ------------------------------------------------
# 4) ⭐ NEW — 보조 성분 필터링 기능 추가
# ------------------------------------------------
def filter_non_base_ingredients(ingredients):
    if ingredients is None:
        return []

    if isinstance(ingredients, str):
        ingredients = [s.strip() for s in ingredients.split(",") if s.strip()]
    else:
        ingredients = list(ingredients)

    active_df = df_ingredients[df_ingredients["종류"] != "보조 성분"]
    active_set = set(active_df["성분명"].astype(str))

    filtered = [ing for ing in ingredients if ing in active_set]
    return filtered


# ------------------------------------------------
# 5) ⭐ NEW — 성분 가이드라인 기능 추가
# ------------------------------------------------
def get_ingredient_guides(ingredients):
    if ingredients is None:
        return []

    if isinstance(ingredients, str):
        ing_list = [s.strip() for s in ingredients.split(",") if s.strip()]
    else:
        ing_list = [str(s).strip() for s in list(ingredients) if str(s).strip()]

    sub = df_ingredients[df_ingredients["성분명"].isin(ing_list)]

    if "성분 가이드" not in sub.columns:
        return []

    sub = sub[["성분명", "성분 가이드"]].dropna(subset=["성분 가이드"])

    guides = []
    for _, row in sub.iterrows():
        guides.append({
            "성분": str(row["성분명"]),
            "가이드": str(row["성분 가이드"]).strip()
        })

    return guides
