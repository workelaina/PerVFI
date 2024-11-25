import sys
import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as TF
from build_models import build_model

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def toTensor(x):
    # x: List of numpy array / A numpy array
    # out: List of torch tensor / A torch tensor
    if isinstance(x, (list, tuple)):
        return list(map(lambda x: TF.to_tensor(x)[None], x))
    return TF.to_tensor(x)[None]


def toArray(x):
    # x: List of torch tensor / A torch tensor
    # out: List of numpy array / A numpy array
    if isinstance(x, (list, tuple)):
        return [np.array(TF.to_pil_image(y[0])) for y in x]
    return np.array(TF.to_pil_image(x[0]))


# l = cv2.imread('test/133.png', cv2.IMREAD_UNCHANGED)
# r = cv2.imread('test/253.png', cv2.IMREAD_UNCHANGED)

video = 'test/Elysia1.png'
capture = cv2.VideoCapture(video)

for i in range(20):
    succ, img = capture.read()

succ, l = capture.read()
cv2.imwrite('result/1.png', l)
succ, r = capture.read()
cv2.imwrite('result/3.png', r)

l = toTensor(l)
r = toTensor(r)
print(l.size(), r.size())

model, infer = build_model('RAFT+PerVFI')
res = infer(l.to('cuda'), r.to('cuda'))
print(res.size())
# cv2.imwrite('result/res.png', toArray(res))
cv2.imwrite('result/2.png', toArray(res))
