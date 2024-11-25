import cv2
import torch
from torch import Tensor
from build_models import build_model

l = cv2.imread('test/133.png', cv2.IMREAD_UNCHANGED)
r = cv2.imread('test/253.png', cv2.IMREAD_UNCHANGED)

# video = 'test/Elysia1.png'
# capture = cv2.VideoCapture(video)

# for i in range(20):
#     succ, img = capture.read()

# succ, l = capture.read()
# succ, r = capture.read()

l = torch.Tensor(l)
r = torch.Tensor(r)
print(l.size(), r.size())

model, infer = build_model('none+PerVFI')
res = infer(l, r)
cv2.imwrite('test/res.png', res.numpy())
