import sys
import cv2
import torch
from torch import Tensor
from torchvision.transforms import functional as TF
from build_models import build_model


def toTensor(x):
    # x: List of numpy array / A numpy array
    # out: List of torch tensor / A torch tensor
    if isinstance(x, (list, tuple)):
        return list(map(lambda x: TF.to_tensor(x)[None], x))
    return TF.to_tensor(x)[None]


l = cv2.imread('test/133.png', cv2.IMREAD_UNCHANGED)
r = cv2.imread('test/253.png', cv2.IMREAD_UNCHANGED)

# video = 'test/Elysia1.png'
# capture = cv2.VideoCapture(video)

# for i in range(20):
#     succ, img = capture.read()

# succ, l = capture.read()
# succ, r = capture.read()

l = toTensor(l)
r = toTensor(r)
print(l.size(), r.size())

model, infer = build_model('RAFT+PerVFI')
res = infer(l, r)
cv2.imwrite('test/res.png', res.numpy())
