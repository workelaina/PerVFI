import cv2
from build_models import build_model

l = cv2.imread('test/133.png', cv2.IMREAD_UNCHANGED)
r = cv2.imread('test/253.png', cv2.IMREAD_UNCHANGED)

print(l.size(), r.size())

# video = 'test/Elysia1.png'
# capture = cv2.VideoCapture(video)

# for i in range(20):
#     succ, img = capture.read()

# succ, l = capture.read()
# succ, r = capture.read()

model, infer = build_model('none+PerVFI')
res = infer(l, r)
cv2.imwrite('test/res.png', res)
