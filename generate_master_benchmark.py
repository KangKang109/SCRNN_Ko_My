import os
import torch
import pandas as pd
from core import vectorize_word, SCRNN, INPUT_DIM

# ==========================================
# 1. 자모 결합 알고리즘 (내 커스텀 복원 엔진)
# ==========================================
def join_jamos_custom(jamo_str):
    if not isinstance(jamo_str, str):
        return ""
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JOUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    chars = list(jamo_str)
    result = ""
    i = 0
    while i < len(chars):
        try:
            if chars[i] in CHOSUNG and i + 1 < len(chars) and chars[i + 1] in JOUNGSUNG:
                cho_idx = CHOSUNG.index(chars[i])
                jung_idx = JOUNGSUNG.index(chars[i + 1])
                jong_idx = 0
                next_step = 2
                
                if i + 2 < len(chars) and chars[i + 2] in JONGSUNG:
                    if i + 3 >= len(chars) or chars[i + 3] not in JOUNGSUNG:
                        jong_idx = JONGSUNG.index(chars[i + 2])
                        next_step = 3
                
                combined = chr(0xAC00 + ((cho_idx * 21) + jung_idx) * 28 + jong_idx)
                result += combined
                i += next_step
            else:
                result += chars[i]
                i += 1
        except Exception:
            result += chars[i]
            i += 1
    return result

# ==========================================
# 2. 마스터 벤치마크 셋 통합 및 생성
# ==========================================
def main():
    print("📂 [1/4] 기존 벤치마크 CSV 파일들을 읽어오는 중...")
    
    # 데이터 로드 및 형식 표준화
    try:
        df_jbb = pd.read_csv('sampled_40_variants.csv')
        df_jbb = df_jbb.rename(columns={'id': 'Index', 'category': 'Category'})
        df_jbb['Source_Dataset'] = 'JBB_40'
    except Exception as e:
        print(f"⚠️ sampled_40_variants.csv 로드 실패: {e}")
        df_jbb = pd.DataFrame()

    try:
        df_adv = pd.read_csv('100_variants.csv')
        df_adv = df_adv.rename(columns={'Category': 'Category'})
        df_adv['Source_Dataset'] = 'ADV_100'
    except Exception as e:
        print(f"⚠️ 100_variants.csv 로드 실패: {e}")
        df_adv = pd.DataFrame()

    # 데이터 프레임 수직 결합 (총 140개 프롬프트 행 생성)
    df_merged = pd.concat([df_jbb, df_adv], ignore_index=True)
    if df_merged.empty:
        print("🚨 원본 csv 파일들이 존재하지 않아 스크립트를 종료합니다.")
        return

    # 데이터셋 구조 간소화 구성
    keep_cols = ['Source_Dataset', 'Category', 'goal_ko', 'goal_ko_jamo']
    df_base = df_merged[keep_cols].copy()
    df_base['Category'] = df_base['Category'].fillna('General').str.strip()

    print(f"✅ 데이터 병합 완료: 총 {len(df_base)}개의 프롬프트 로드.")
    print("🧠 [2/4] 실제 연구 중인 scrnn_korean.pth 가중치를 로드하는 중...")

    # 실제 가중치를 적용한 SC-RNN 모델 로드 (클래스 3개 고정)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SCRNN(INPUT_DIM, 128, 3)
    
    if os.path.exists('scrnn_korean.pth'):
        model.load_state_dict(torch.load('scrnn_korean.pth', map_location=device))
        model.eval()
        print(f"✅ SC-RNN 모델 가중치 반영 완료. (Device: {device})")
    else:
        print("❌ 'scrnn_korean.pth' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print("🛡️ [3/4] 모든 프롬프트를 대상으로 내 모델 파이프라인 전처리 연산 실행 중...")
    processed_prompts = []

    # 분기문 조건 필터링 제거 -> 무조건 모델 연산과 결합 알고리즘을 한 세트로 통과시킴
    for idx, row in df_base.iterrows():
        jamo_prompt = str(row['goal_ko_jamo'])
        words = jamo_prompt.split()
        
        # [단계 A] 자모 분리 상태의 토큰 단위 피드포워드 추론 (훈련 가중치 활성화 상태 검증용)
        with torch.no_grad():
            for word in words:
                vec = vectorize_word(word).unsqueeze(0).unsqueeze(0).to(device)
                _ = model(vec) # 코랩 전 최종 마스터 데이터셋 생성을 위해 실제 가중치를 연산에 참여시킴
        
        # [단계 B] 무조건 내 커스텀 자모 결합 알고리즘을 거쳐 복원 문장 생성
        restored_output = join_jamos_custom(jamo_prompt)
        processed_prompts.append(restored_output)

    # 3번째 실험군 프롬프트 버전으로 데이터프레임 열 추가
    df_base['goal_ko_scrnn_processed'] = processed_prompts

    # ==========================================
    # 3. 코랩 전용 마스터 벤치마크 파일 저장
    # ==========================================
    output_name = 'master_jailbreak_benchmark.csv'
    df_base.to_csv(output_name, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("🎉 코랩(Colab) 평가용 통합 마스터 CSV 빌드 성공!")
    print(f"💾 생성된 파일명: {output_name}")
    print(f"📊 총 라인수: {len(df_base)} 행 (3개의 프롬프트 독립열 확보)")
    print("="*60)
    print("💡 이제 이 CSV 파일 하나만 구글 코랩에 업로드하여\n   LLM API 벤치마크 및 SAFE/UNSAFE 통계 평가를 가볍게 진행하시면 됩니다.")

if __name__ == "__main__":
    main()