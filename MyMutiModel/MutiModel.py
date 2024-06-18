import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, ViTModel, ViTFeatureExtractor
import torch.nn.functional as F

class MultiModalModel(nn.Module):
    def __init__(self, vit_model, bert_model, hidden_dim):
        super(MultiModalModel, self).__init__()
        self.vit_model = vit_model
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(self.vit_model.config.hidden_size + self.bert_model.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)  # 二分类输出

    def forward(self, image, input_ids, attention_mask):
        # 图像特征提取
        image_features = self.vit_model(pixel_values=image).last_hidden_state[:, 0, :]  # 取CLS token特征

        # 文本特征提取
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        text_features = self.bert_model(**inputs).last_hidden_state[:, 0, :]  # 取CLS token特征

        # 特征融合
        combined_features = torch.cat((image_features, text_features), dim=1)

        # 全连接层
        x = self.fc(combined_features)
        x = self.relu(x)
        x = self.output(x)
        return x

# 初始化模型
vit_model = ViTModel.from_pretrained('../ViTProcessor')
bert_model = BertModel.from_pretrained('../bert')
model = MultiModalModel(vit_model, bert_model, hidden_dim=512)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
