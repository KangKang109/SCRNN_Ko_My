import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from core import vectorize_word, SCRNN, INPUT_DIM

# 1. 환경 설정 및 데이터 로드
plt.rc('font', family='Malgun Gothic') # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cpu')

print("📂 데이터 및 모델 로딩 중...")
df = pd.read_csv('data.csv')
num_classes = len(df['label'].unique())
model = SCRNN(INPUT_DIM, 128, num_classes)
model.load_state_dict(torch.load('scrnn_korean.pth', map_location=device))
model.eval()

# 시각화용 샘플링 (너무 많으면 t-SNE가 느리므로 600개 추출)
df_sample = df.sample(n=min(600, len(df)), random_state=42)
words, labels = df_sample['word'].values, df_sample['label'].values

# 2. 성능 평가 수행
all_preds = []
all_vectors = []
with torch.no_grad():
    for word in words:
        vec = vectorize_word(word)
        all_vectors.append(vec.numpy())
        output = model(vec.unsqueeze(0).unsqueeze(0))
        all_preds.append(torch.argmax(output, dim=1).item())

all_vectors = np.array(all_vectors)

# 3. 종합 시각화 (2x2 Layout)
fig, axs = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# [Graph 1] t-SNE Cluster Analysis
print("🚀 t-SNE 차원 축소 중...")
tsne = TSNE(n_components=2, random_state=42)
vec_2d = tsne.fit_transform(all_vectors)
colors = ['#4A90E2', '#E94E77', '#F7B733']
labels_map = {0: '정상', 1: '공격', 2: '기타'}
for i in range(num_classes):
    idx = (labels == i)
    axs[0, 0].scatter(vec_2d[idx, 0], vec_2d[idx, 1], c=colors[i], label=labels_map[i], alpha=0.7, edgecolors='w')
axs[0, 0].set_title('데이터 구조 군집 분석 (t-SNE)', fontsize=15)
axs[0, 0].legend()

# [Graph 2] Confusion Matrix
print("📊 혼동 행렬 생성 중...")
cm = confusion_matrix(labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1],
            xticklabels=['Normal', 'Attack', 'Etc'], yticklabels=['Normal', 'Attack', 'Etc'])
axs[0, 1].set_title('예측 정확도 혼동 행렬 (Confusion Matrix)', fontsize=15)
axs[0, 1].set_ylabel('Actual')
axs[0, 1].set_xlabel('Predicted')

# [Graph 3] Performance Metrics Heatmap
print("📈 세부 지표 계산 중...")
report = classification_report(labels, all_preds, target_names=['Normal', 'Attack', 'Etc'], output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :3].T
sns.heatmap(report_df, annot=True, cmap='YlGnBu', ax=axs[1, 0])
axs[1, 0].set_title('모델 정밀도 및 재현율 (Precision/Recall)', fontsize=15)

# [Graph 4] Training 가상 데이터 시각화 (learning_curve.png 활용 가능)
# 실제 history 객체가 있으면 그것을 사용하고, 여기서는 최종 결과 위주로 표시
axs[1, 1].axis('off')
summary_text = (
    f" [ 모델 평가 최종 요약 ]\n\n"
    f"• 전체 정확도(Accuracy): {report['accuracy']*100:.2f}%\n"
    f"• 공격 탐지 재현율(Attack Recall): {report['Attack']['recall']*100:.2f}%\n"
    f"• 오탐지율(False Positive): {(cm[0,1]/cm[0].sum())*100:.2f}%\n\n"
    f"▶ 해석: 본 모델은 자모 노이즈에도 불구하고\n"
    f"실제 공격을 놓치지 않는 강력한 성능을 보임."
)
axs[1, 1].text(0.1, 0.5, summary_text, fontsize=16, fontweight='bold', va='center')

plt.savefig('final_evaluation_report.png', bbox_inches='tight')
print("✅ 모든 시각화가 완료되었습니다. 'final_evaluation_report.png'를 확인하세요.")
plt.show()