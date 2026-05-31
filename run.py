import torch
import pandas as pd
from core import vectorize_word, SCRNN, INPUT_DIM

# [독립 구현] 외부 라이브러리 없이 한글 자모를 결합하는 함수
def join_jamos_custom(jamo_str):
    """
    분리된 자모(예: ㅍㅔㅇㅣㄹㅗㄷㅡ)를 완성형 한글(예: 페이로드)로 결합합니다.
    """
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JOUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    chars = list(jamo_str)
    result = ""
    i = 0
    while i < len(chars):
        try:
            # 현재 문자가 초성이고 다음 문자가 중성인 경우 결합 시도
            if chars[i] in CHOSUNG and i + 1 < len(chars) and chars[i + 1] in JOUNGSUNG:
                cho_idx = CHOSUNG.index(chars[i])
                jung_idx = JOUNGSUNG.index(chars[i + 1])
                jong_idx = 0
                next_step = 2
                
                # 종성 확인
                if i + 2 < len(chars) and chars[i + 2] in JONGSUNG:
                    # 다음 글자가 중성이라면 현재 자음은 다음 글자의 초성이어야 함
                    if i + 3 >= len(chars) or chars[i + 3] not in JOUNGSUNG:
                        jong_idx = JONGSUNG.index(chars[i + 2])
                        next_step = 3
                
                # 한글 유니코드 공식: (초성 * 21 + 중성) * 28 + 종성 + 0xAC00
                combined = chr(0xAC00 + ((cho_idx * 21) + jung_idx) * 28 + jong_idx)
                result += combined
                i += next_step
            else:
                # 결합 조건이 안 맞으면 그대로 유지
                result += chars[i]
                i += 1
        except Exception:
            result += chars[i]
            i += 1
    return result

# 1. 클래스 라벨 자동 로드 (data.csv 기준)
try:
    df = pd.read_csv('data.csv')
    num_classes = len(df['label'].unique())
except Exception as e:
    print(f"❌ data.csv 파일을 읽을 수 없습니다: {e}")
    exit()

# 2. 모델 로드
model = SCRNN(INPUT_DIM, 128, num_classes)
try:
    # 맵핑 정보를 위해 weights_only=True 권장
    model.load_state_dict(torch.load('scrnn_korean.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ 학습된 보안 모델을 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"❌ 모델 파일(.pth) 로드 실패: {e}")
    print("train.py를 실행하여 모델을 먼저 생성했는지 확인하세요.")
    exit()

print("\n" + "="*50)
print("🛡️  동국대학교 종합설계 - SC-RNN 보안 탐지 시스템")
print("="*50)
print("테스트할 단어를 자모음 단위로 입력하세요. (예: ㅍㅔㅇㅣㄹㅗㄷㅡ)")
print("종료하려면 'exit'를 입력하세요.")

while True:
    user_input = input("\n입력값 >> ").strip()
    
    if user_input.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break
    
    if not user_input:
        continue

    # [핵심 로직] 풀어진 자모음을 하나의 단어로 결합 (직접 구현한 함수 사용)
    try:
        recognized_word = join_jamos_custom(user_input)
    except Exception:
        recognized_word = user_input 

    with torch.no_grad():
        # 1. 입력값 벡터화 (core.py 내 함수 사용)
        vec = vectorize_word(user_input).unsqueeze(0).unsqueeze(0)
        
        # 2. 모델 추론
        output = model(vec)
        
        # 3. 확률 및 클래스 계산
        pred = torch.argmax(output, dim=1).item()
        prob = torch.nn.functional.softmax(output, dim=1)[0][pred] * 100
        
        # 4. 결과 출력
        print(f"🤖 모델 인식: 입력하신 '{user_input}'을(를) '{recognized_word}'(으)로 인식했습니다.")
        print(f"🔍 탐지 결과: Class {pred} (확신도: {prob:.2f}%)")
        
        if pred == 1: # 보통 1을 공격/위험으로 설정하는 경우
            print("⚠️ 경고: 보안 위협 패턴이 감지되었습니다!")
        else:
            print("✅ 정상적인 입력 패턴입니다.")

print("="*50)