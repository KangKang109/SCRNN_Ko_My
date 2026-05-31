import torch
import torch.nn as nn
from jamo import h2j, j2hcj

# 한국어 자모 설정 (66개)
JAMOS = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢㄳㄴㅈㄴㅎㄹㄱㄹㅁㄹㅂㄹㅅㄹㅌㄹㅍㄹㅎㅂㅅ'
jamo2idx = {j: i for i, j in enumerate(JAMOS)}
INPUT_DIM = len(JAMOS) * 3  # SC-RNN (Start, Internal, End) -> 198차원

# core.py 내 vectorize_word 함수 수정
def vectorize_word(word):
    feat = torch.zeros(3, len(JAMOS))
    try:
        jamos = list(j2hcj(h2j(str(word))))
        if len(jamos) < 1: return feat.view(-1)
        
        # 1. Start Character
        if jamos[0] in jamo2idx: feat[0, jamo2idx[jamos[0]]] = 1
        # 2. End Character
        if jamos[-1] in jamo2idx: feat[1, jamo2idx[jamos[-1]]] = 1
        
        # 3. Internal Bag-of-characters (정규화 추가)
        if len(jamos) > 2:
            internal_jamos = jamos[1:-1]
            for j in internal_jamos:
                if j in jamo2idx: 
                    feat[2, jamo2idx[j]] += 1
            # 내부 자모 개수로 나누어 평균 빈도로 변환 (Scaling)
            feat[2] = feat[2] / len(internal_jamos)
            
    except: pass
    return feat.view(-1)

class SCRNN(nn.Module):
    """SC-RNN 아키텍처 (논문 구조 유지)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SCRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])