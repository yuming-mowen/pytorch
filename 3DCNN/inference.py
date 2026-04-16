import numpy as np
import torch
import cv2
import C3D_model
from torchsummary import summary

# 视频帧切片函数
def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def inference():
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将标签文档读入
    with open("./data/labels.txt", 'r') as f:
        class_names = f.readlines()
        f.close()

    model = C3D_model.C3D(num_classes=101)
    # 加载模型参数到model中
    checkpoint = torch.load('./model_result/models/C3D_epoch-25.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # 将模型放入设备中
    model.to(device)
    model.eval()

    # 读取数据
    video = "./CXK.mp4"
    cap = cv2.VideoCapture(video)
    retaining = True  # 标志视频是否结束的标志

    clip = []
    # 读取视频所有帧
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        # 视频处理
        tmp_ = cv2.resize(frame, (171, 128))
        # 将视频剪切为112*112
        tmp_ = center_crop(tmp_)
        # 视频帧归一化
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        # 每16帧组合
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            # 维度交换
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            # 进行模型推理
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            # 取出对应的标签
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            # 在视频上写入标签
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            # 弹出第一帧
            clip.pop(0)  # 形成滑动窗口
        # 展示视频
        cv2.imshow("result", frame)
        cv2.waitKey(30)
    # 释放空间
    cap.release()
    cv2.destroyWindow()


if __name__ == "__main__":
    inference()