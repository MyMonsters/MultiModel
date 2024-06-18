from sklearn.metrics import accuracy_score, f1_score
import torch
from MutiModel import model
from Dataset import dataloader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
evaluate_model(model, dataloader)
