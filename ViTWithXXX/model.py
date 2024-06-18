import torch
import torch.nn as nn
from attention.SE import SE
from attention.CBAM import CBAM
from attention.SimAM import Simam_module
class ViTWithXXX(nn.Module):
    def __init__(self, vit_model, se_block):
        super(ViTWithXXX, self).__init__()
        self.vit_model = vit_model
        self.se_block = SE(channel=3)
        # self.cbam_block = CBAM(3)
        self.SimAM_block = Simam_module(3)

    def forward(self, pixel_values):
        print(pixel_values.shape)
        b, c, h ,w = pixel_values.shape;
        # x = pixel_values.view(1, 64, 32, 32)
        # x = self.se_block(x)
        # print(x.shape)
        # x = x.view(1, 64, 1024)
        x = self.se_block(pixel_values)
        print(x.shape)
        # x = x.view(1, 64, 1024)
        # x = x.view(b, h*w, c)
        x = self.SimAM_block(x)
        x = self.vit_model(x)
        # last_hidden_state = x.last_hidden_state

        # 获取批次大小和隐藏状态的大小
        # batch_size, seq_length, hidden_size = last_hidden_state.shape
        # print("seq_length",seq_length)
        # # 将last_hidden_state重塑为4D张量 (batch_size, hidden_size, height, width)
        # x = last_hidden_state.view(batch_size, hidden_size, int(seq_length ** 0.5), int(seq_length ** 0.5))

        # 通过SE模块


        # 将张量调整回原始形状
        # x = x.view(batch_size, hidden_size, seq_length).permute(0, 2, 1).contiguous()
        # x = x.last_hidden_state[:, 0, :].squeeze()
        # x = x.view(1, 768, 32, 32)
        # x = self.se_block(x)
        # x = x.view(1, 64, 1024)
        # print(x)
        # print(pixel_values.shape)
        # last_hidden_state = outputs.last_hidden_state
        # batch_size, seq_length, hidden_size = last_hidden_state.shape
        # x = last_hidden_state.permute(0, 2, 1).contiguous()
        # 通过SE模块


        # 调整回原来的形状
        # x = x.view(batch_size, hidden_size, seq_length).permute(0, 2,
        #                                                         1).contiguous()  # [batch_size, seq_length, hidden_size]
        # attention_output = self.SEattention(last_hidden_state)
        return x
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 随机生成一个输入特征图
    se = SE(channel=512, reduction=8)  # 实例化SE模块，设置降维比率为8
    output = se(input)  # 将输入特征图通过SE模块进行处理
    print(output.shape)  # 打印处理后的特征图形状，验证SE模块的作用