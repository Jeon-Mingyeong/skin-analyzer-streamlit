import streamlit as st
from scoring import full_pipeline

# 페이지 설정
st.set_page_config(page_title="피부 분석 웹앱")

# --------------------------
# CSS 커스텀 (정민 UI 그대로 구현)
# --------------------------
st.markdown("""
<style>

/* 전체 배경 */
body {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: #fafafa;
}

/* Streamlit 기본 요소 여백 제거 */
.block-container {
    padding-top: 20px !important;
    padding-bottom: 20px !important;
}

/* 카드 스타일 */
.custom-box {
    background: white;
    padding: 30px 26px;
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
}

/* 입력 필드 스타일 */
textarea, input, select {
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* 버튼 스타일 */
.stButton>button {
    width: 100%;
    background: rgb(227,223,53);
    color: black;
    font-weight: 700;
    font-size: 16px;
    border-radius: 10px;
    height: 48px;
    border: none;
}

.stButton>button:hover {
    background: rgb(210,206,50);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# 화면 구조 시작
# --------------------------------

st.markdown("""
<div style="text-align:center; margin-bottom:8px; font-size:13px; color:#444;">
이 제품 내 피부에게 최선일까? 🤔
</div>

<div style="text-align:center; margin-bottom:26px; font-size:22px; font-weight:700; line-height:1.4;">
피부 타입과 고민을 기반으로<br>진단해 드려요!
</div>
""", unsafe_allow_html=True)

# --------------------------------
# 입력 UI 카드 박스
# --------------------------------
st.markdown('<div class="custom-box">', unsafe_allow_html=True)

nickname = st.text_input("닉네임", placeholder="ex. 김슈니")

skin_type = st.selectbox(
    "피부 타입",
    ["지성", "복합성", "건성", "민감성"],
)

concern = st.text_area(
    "피부 고민",
    placeholder="피부 고민을 자유롭게 입력해 주세요!\nex. 하루 종일 푸석거려요, 세로모공 고민이에요 등"
)

ingredients = st.text_area(
    "전성분 목록",
    placeholder="구매한 제품의 전성분 목록을 복사해 붙여주세요!\n"
                "(제품정보 제공고시 → 화장품 전 성분)\n\n"
                "ex. 정제수, 글리세린, 토코페롤..."
)

submit = st.button("적합도 알아보기")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------
# 분석 버튼 동작
# --------------------------------
if submit:
    if not concern.strip() or not ingredients.strip():
        st.warning("피부 고민과 전성분을 모두 입력해주세요!")
    else:
        with st.spinner("분석 중입니다... ⏳"):
            result = full_pipeline(
                concern,
                ingredients,
                ["지성","복합성","건성","민감성"].index(skin_type) + 1
            )

        st.success("분석 완료! 😊")

        st.markdown("---")
        st.write("### ✔ 최종 점수:", result.get("최종점수"))
        st.write("### ✔ 피부 타입:", result.get("피부타입"))
        st.write("### ✔ 예측 고민:", result.get("예측고민"))
        st.write("### 📘 성분 가이드:")
        st.write(result.get("성분가이드"))
