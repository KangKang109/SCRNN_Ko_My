import pandas as pd
import random
from jamo import h2j, j2hcj

# 1. 핵심 키워드 정의 (MITRE ATT&CK 및 침해사고 대응 키워드 기반)
# Label 0: 정상 시스템 용어 / Label 1: 위험 보안 위협 / Label 2: 일반 서비스 이용
keywords = {
    0: [
        "관리자", "어드민", "로그인", "접속", "데이터베이스", "서버", "설정", "시스템",
        "환경설정", "네트워크", "업데이트", "동기화", "인증", "승인", "계정관리", "대시보드"
    ],
    1: [
        "해킹", "공격", "페이로드", "취약점", "탈취", "스크립트", "악성코드", "우회",
        "리버스셸", "권한상승", "백도어", "무차별대입", "인젝션", "데이터유출", "바이러스", "루트킷",
        "익스플로잇", "트로이목마", "랜섬웨어", "정보수집", "암호해독", "키로깅", "명령제어"
    ],
    2: [
        "사용자", "게시판", "프로필", "검색", "채팅", "목록", "알림", "도움말",
        "커뮤니티", "작성하기", "마이페이지", "문의사항", "공지사항", "이벤트", "쪽지", "친구추가"
    ]
}

def generate_typo(word):
    """한국어 자모 단위 노이즈 생성 함수 (SC-RNN 학습용)"""
    jamos = list(j2hcj(h2j(word)))
    if len(jamos) < 3: return "".join(jamos)
    
    case = random.randint(1, 5)
    
    # 전략: 첫 글자와 마지막 글자의 뼈대는 어느 정도 유지하며 중간을 흔듦
    if case == 1 and len(jamos) > 3: # 중간 자모 순서 바꾸기
        idx = random.randint(1, len(jamos)-2)
        jamos[idx], jamos[idx+1] = jamos[idx+1], jamos[idx]
    elif case == 2: # 중간 자모 중복 삽입
        idx = random.randint(1, len(jamos)-2)
        jamos.insert(idx, jamos[idx])
    elif case == 3: # 중간 자모 삭제
        idx = random.randint(1, len(jamos)-2)
        jamos.pop(idx)
    elif case == 4: # 유사 자음 교체 (예: ㅂ -> ㅃ, ㄱ -> ㄲ)
        replace_map = {'ㄱ':'ㄲ', 'ㄴ':'ㄹ', 'ㄷ':'ㄸ', 'ㅂ':'ㅃ', 'ㅅ':'ㅆ', 'ㅈ':'ㅉ', 'ㅏ':'ㅑ', 'ㅓ':'ㅕ'}
        idx = random.randint(0, len(jamos)-1)
        if jamos[idx] in replace_map:
            jamos[idx] = replace_map[jamos[idx]]
    
    # case 5는 원본 자모 나열을 그대로 반환 (정상 샘플)
    return "".join(jamos)

# 2. 대량 데이터 생성 (변형 데이터 200개로 확장하여 일반화 성능 향상)
dataset = []
for label, words in keywords.items():
    for word in words:
        # 원본 추가
        dataset.append({"word": "".join(list(j2hcj(h2j(word)))), "label": label})
        # 변형 데이터 생성
        for _ in range(200):
            dataset.append({"word": generate_typo(word), "label": label})

df = pd.DataFrame(dataset)
# 중복 데이터 제거 (우연히 같은 변형이 생길 수 있음)
df = df.drop_duplicates().sample(frac=1).reset_index(drop=True)

df.to_csv('data.csv', index=False, encoding='utf-8-sig')
print(f"✅ 총 {len(df)}개의 보안 특화 자모 노이즈 데이터셋 생성 완료!")