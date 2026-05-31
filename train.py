# train.py 수정
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import random # 추가
from sklearn.model_selection import train_test_split # 추가
from core import vectorize_word, SCRNN, INPUT_DIM

# 1. 데이터 로드 및 분리
try:
    df = pd.read_csv('data.csv')
except:
    df = pd.read_csv('data.csv', encoding='cp949')

# 데이터를 리스트 형태로 변환 후 셔플
data_list = df.to_dict('records')
random.shuffle(data_list)

# 학습(80%)과 검증(20%) 데이터로 분리
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

num_classes = len(df['label'].unique())
model = SCRNN(INPUT_DIM, 128, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

history = {'loss': [], 'acc': [], 'val_acc': []}

for epoch in range(100): # 에포크는 100회 정도면 충분합니다
    model.train()
    epoch_loss, correct = 0, 0
    
    for item in train_data:
        word, label = item['word'], item['label']
        inputs = vectorize_word(word).unsqueeze(0).unsqueeze(0)
        target = torch.tensor([label])
        
        output = model(inputs)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if torch.argmax(output, dim=1).item() == label: correct += 1
    
    # --- 검증 단계 (Validation) ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for item in val_data:
            inputs = vectorize_word(item['word']).unsqueeze(0).unsqueeze(0)
            output = model(inputs)
            if torch.argmax(output, dim=1).item() == item['label']: val_correct += 1
    
    history['loss'].append(epoch_loss / len(train_data))
    history['acc'].append(correct / len(train_data))
    history['val_acc'].append(val_correct / len(val_data))
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {history['loss'][-1]:.4f} | Train Acc: {history['acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")

torch.save(model.state_dict(), 'scrnn_korean.pth')
print("✅ 검증 완료 및 모델 저장 성공!")