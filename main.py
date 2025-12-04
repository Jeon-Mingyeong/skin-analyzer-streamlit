

from scoring import full_pipeline

print("===== 피부타입 선택 =====")
print("1) 지성  2) 복합성  3) 건성  4) 민감성")
skin = int(input("번호 입력: "))

review = input("\n고민 입력: ")
ings = input("\n전성분 입력: ")

r = full_pipeline(review, ings, skin)

print("\n======= 결과 =======")
for k,v in r.items():
    print(f"{k} : {v}")

# 🔹 성분 가이드는 따로 보기 좋게 출력
guides = r.get("성분가이드")
if guides:
    print("\n===== 성분 가이드 =====")
    for item in guides:
        name = item.get("성분")
        guide = item.get("가이드")
        print(f"\n● {name}\n   → {guide}")