import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from MutiModel import model
from Dataset import dataloader,validDataloader
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 训练模型
num_epochs = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader):
        # 获取数据
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 前向传播
        outputs = model(images, input_ids, attention_mask)

        # 计算标签，这里简单假设所有样本的标签为1
        labels = torch.ones(outputs.size(0), 1).to(device)

        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

print('Finished Training')
from sklearn.metrics import accuracy_score, f1_score
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(images, input_ids, attention_mask)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend([1] * len(predictions))  # 假设所有标签都为1

    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])

    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')


# 使用验证数据集评估模型
evaluate_model(model, validDataloader)