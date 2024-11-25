import cv2
from build_models import build_model

l = cv2.imread('test/133.png')
r = cv2.imread('test/253.png')

model, infer = build_model('none+PerVFI')

cv2.imsave(infer(l, r), 'test/res.png')
