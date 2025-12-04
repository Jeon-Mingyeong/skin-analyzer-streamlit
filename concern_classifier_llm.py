import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax
import torch


# ==========================================
# 0) 텍스트 전처리
# ==========================================
def clean_text(t: str) -> str:
    t = str(t)
    t = t.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"[^가-힣0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ==========================================
# 1) SBERT 로딩
# ==========================================
sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


# ==========================================
# 2) LLM 로딩 (Polyglot-ko 1.3b, 랜덤 OFF)
# ==========================================
model_id = "EleutherAI/polyglot-ko-1.3b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
llm = llm.to(device)


def ask_llm(prompt: str, max_new_tokens=70):
    """ 의미 보존 + 동의어 확장 (랜덤 OFF) """
    inputs = tokenizer(prompt, return_tensors="pt")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ==========================================
# 3) DB1 로딩
# ==========================================
df = pd.read_excel("data/skin_concerns_canon.xlsx")

text_col = "소비자 언어 (리뷰)"
label_col = "피부고민"

labels = df[label_col].unique().tolist()

label_texts = {lbl: [] for lbl in labels}

for _, row in df.iterrows():
    cleaned = clean_text(row[text_col])
    if cleaned:
        label_texts[row[label_col]].append(cleaned)


# ==========================================
# 4) SBERT 라벨별 개별 벡터 저장
# ==========================================
label_vectors = {}
for lbl in labels:
    embs = sbert.encode(label_texts[lbl], convert_to_numpy=True)
    label_vectors[lbl] = embs


# ==========================================
# 5) SBERT top-k 평균 유사도 계산
# ==========================================
def sbert_score(review_emb, vectors, top_k=5):
    sims = []
    for v in vectors:
        sim = np.dot(review_emb, v) / (
            np.linalg.norm(review_emb) * np.linalg.norm(v)
        )
        sims.append(sim)

    sims = sorted(sims, reverse=True)
    return float(sum(sims[:top_k]) / top_k)


# ==========================================
# 6) 전문가 규칙
# ==========================================
expert_rules = {
    "수분부족": ["속당김", "건조", "푸석", "당김"],
    "미백/잡티": ["미백", "화이트닝", "톤업", "밝아짐", "흉터", "자국", "잡티", "기미"],
    "피지/블랙헤드": ["기름", "번들", "유분", "피지"],
    "여드름": ["트러블", "뾰루지", "염증"],
    "모공": ["모공", "구멍"],
    "각질": ["각질", "일어남"],
    "홍조": ["홍조","붉어짐","홍당무","빨개짐"],
    "주름/탄력": ["탄력","주름","리프팅","처짐","늘어짐","탄탄","탱탱"]
}


# ==========================================
# 7) 핵심 키워드 자동 추출 + 확장문 프롬프트
# ==========================================
def extract_keywords(review: str):
    keys = []
    for lbl, kws in expert_rules.items():
        for kw in kws:
            if kw in review:
                keys.append(kw)
    return list(set(keys))


def build_prompt(review: str):
    keys = extract_keywords(review)
    key_text = ", ".join(keys) if keys else ""

    if keys:
        force_kw = f"핵심 단어({key_text})는 반드시 포함하고 "
    else:
        force_kw = ""

    return (
        f"다음 문장을 의미는 절대 바꾸지 말고, "
        f"{force_kw}"
        f"동의어나 자연스러운 표현을 추가해서 길게 확장해줘: {review}"
    )


# ==========================================
# 8) 고민 예측 (SBERT + 전문가 규칙)
# ==========================================
def predict_concern(review: str, top_k=2):
    emb = sbert.encode([review], convert_to_numpy=True)[0]

    raw_scores = {}
    for lbl in labels:
        raw_scores[lbl] = sbert_score(emb, label_vectors[lbl])

    probs = softmax(list(raw_scores.values()))
    score_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # 규칙 가중치
    r = review.lower()
    OVERRIDE = 3.0

    for lbl, kws in expert_rules.items():
        for kw in kws:
            if kw in r:
                score_dict[lbl] += OVERRIDE

    total = sum(score_dict.values())
    norm = {k: v / total for k, v in score_dict.items()}

    final = sorted(norm, key=lambda x: norm[x], reverse=True)[:top_k]
    return final, norm


# ==========================================
# 9) Ensemble (원문 3표 + 확장문 1표)
# ==========================================
def ensemble_predict(review: str):
    votes = []
    prob_list = []

    # 원문 (3표)
    p1, pr1 = predict_concern(review)
    votes += p1 * 3
    prob_list.append(pr1)

    # 확장문 (1표)
    expanded = ask_llm(build_prompt(review))
    p2, pr2 = predict_concern(expanded)
    votes += p2
    prob_list.append(pr2)

    # 최종 투표
    counter = Counter(votes)
    final_label, _ = counter.most_common(1)[0]

    # 확률 평균
    final_probs = {}
    for lbl in labels:
        final_probs[lbl] = float(np.mean([p.get(lbl, 0) for p in prob_list]))

    return final_label, final_probs


# ==========================================
# 10) scoring.py 호환 alias
# ==========================================
df_concerns = df
label_col = label_col
text_col = text_col
