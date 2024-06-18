import pandas as pd
from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
from PIL import Image
import requests
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, dataframe, vit_processor, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_url = self.dataframe.iloc[idx]['image_url']
        caption = self.dataframe.iloc[idx]['caption']

        # 加载图像
        # response = requests.get(image_url)
        # image = Image.open(BytesIO(response.content)).convert('RGB')
        image = Image.open(image_url)
        # 如果图像有透明度通道，则转换为RGBA格式
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        image = self.vit_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # 加载文本
        inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption
        }

# 读取xlsx文件
dataframe = pd.read_excel('./train.xlsx')
validDataframe = pd.read_excel('./valid.xlsx')
# 初始化ViT处理器和BERT分词器
vit_processor = ViTImageProcessor.from_pretrained('../ViTProcessor')
tokenizer = BertTokenizer.from_pretrained('../bert')

# 创建数据集和数据加载器
dataset = MultiModalDataset(dataframe, vit_processor, tokenizer)
validdataset = MultiModalDataset(validDataframe, vit_processor, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
validDataloader = DataLoader(validdataset, batch_size=32, shuffle=True)