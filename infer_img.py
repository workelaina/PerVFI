import sys
import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as TF
from build_models import build_model
from tools import Tools

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model, infer = build_model('RAFT+PerVFI')

# l = cv2.imread('test/133.png', cv2.IMREAD_UNCHANGED)
# r = cv2.imread('test/253.png', cv2.IMREAD_UNCHANGED)
# l = Tools.toTensor(l)
# r = Tools.toTensor(r)
# print(l.size(), r.size())
# res = infer(l.to('cuda'), r.to('cuda'))
# print(res.size())
# cv2.imwrite('result/res.png', toArray(res))

video = 'test/Elysia1.mp4'
capture = cv2.VideoCapture(video)
nb_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
nb_frames = int(nb_frames + 0.5)
ori_fps = capture.get(cv2.CAP_PROP_FPS)
if ori_fps < 0.1:
    raise RuntimeError("The frame rate is %s!" % str(ori_fps))
# duration = nb_frames / ori_fps
print('nb_frames:', nb_frames)
print('ori_fps:', ori_fps)
print()

i1 = None
for i in range(nb_frames):
    print('batch', i)
    succ, i3 = capture.read()
    if not succ:
        raise RuntimeError(i)
    if i:
        cv2.imwrite('result/%d.png' % (i*2-1), Tools.toArray(infer(
            Tools.toTensor(i1).to('cuda'),
            Tools.toTensor(i3).to('cuda')
        )))
    cv2.imwrite('result/%d.png' % (i*2), i3)
    i1 = i3

# ffmpeg -fflags +genpts -r 30 -i raw.h264 -c:v copy output.mp4
