#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
CREATED BY HAORAN
TIME: 2021-06-06

"""
# encoding=utf-8
import sys

sys.path.append('..')
from two_stage_model import UNetStage1 as Net1
from two_stage_model import UNetStage2 as Net2
import numpy as np
from PIL import Image
import torch
import torchvision
import cv2 as cv


def one_picture_port(src):
    torch.cuda.empty_cache()
    global stage1_ouput, output2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # model_path1=r"E:\研究\stage1_5 信息熵.pth"
    # model_path2=r"E:\研究\stage2_09-18加入信息熵loss_checkpoint44-two_stage-0.066889-f10.714166-precision0.611969-acc0.991789-recall0.888501.pth"
    model_path1 = r'./stage1.pth'
    model_path2 = r'./stage2.pth'
    checkpoint1 = torch.load(model_path1, map_location=device)
    checkpoint2 = torch.load(model_path2, map_location=device)
    model1 = Net1().to(device)
    model2 = Net2().to(device)

    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    model1.eval()
    model2.eval()

    # src 为Image格式
    # print(src.split())
    if len(src.split()) != 3:
        src = src.convert('RGB')

    # src = src.resize((src.size[0] // 2, src.size[1] // 2))

    try:
            img = torchvision.transforms.Compose([
                # 对于数据集 标准化
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
            ])(src)
            img = img[np.newaxis, :, :, :].cuda()
            # print(img.size)
            output = model1(img)
            stage1_ouput = output[0].detach()
            model2_input = torch.cat((stage1_ouput, img), 1).detach()
            output2 = model2(model2_input, output[1], output[2], output[3])
            output2[0].detach()
            # 在内存要求内 ，停止循环   break
    except Exception as e:
            print('The error,内存不足!!!改变图像大小到原本的一半')
            # print(img.shape)
            src = src.resize((src.size[0] // 2, src.size[1] // 2))
            # img=img.resize((a//2,b//2))
            # print(model2_input.shape)
            # img = torchvision.transforms.Compose([
            #     torchvision.transforms.Resize((src.size[1] // 2, src.size[0] // 2)),
            #     torchvision.transforms.ToTensor(),
            #     # torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
            # ])(src)
            torch.cuda.empty_cache()
            img = torchvision.transforms.Compose([
                # 对于数据集 标准化
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
            ])(src)
            img = img[np.newaxis, :, :, :].cuda()
            # print(img.size)
            output = model1(img)
            stage1_ouput = output[0].detach()
            model2_input = torch.cat((stage1_ouput, img), 1).detach()
            output2 = model2(model2_input, output[1], output[2], output[3])
            output2[0].detach()
            print('resize:', (src.size[1] // 2, src.size[0] // 2))

    output = np.array(stage1_ouput.cpu().detach().numpy(), dtype='float32')

    output = output.squeeze(0)
    output = np.transpose(output, (1, 2, 0))
    output_ = output.squeeze(2)

    output2 = np.array(output2[0].cpu().detach().numpy(), dtype='float32')
    output2 = output2.squeeze(0)
    output2 = np.transpose(output2, (1, 2, 0))
    output2_ = output2.squeeze(2)

    output = np.array(output_) * 255
    output = np.asarray(output, dtype='uint8')
    output2 = np.array(output2_) * 255
    output2 = np.asarray(output2, dtype='uint8')
    output_cai = cv.applyColorMap(output, cv.COLORMAP_JET)
    output2_cai = cv.applyColorMap(output2, cv.COLORMAP_JET)
    torch.cuda.empty_cache()
    return output, output_cai, output2_cai, output2


if __name__ == '__main__':
    src = Image.open("./test_data/1-1.png")
    _, stage1, stage2, _ = one_picture_port(src)
    cv.imshow("1s", stage2)
    cv.imwrite("1-1st2.bmp", stage2)
    cv.imwrite("1-1st1.bmp", stage1)
    cv.waitKey(0)
