import cv2

cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0)
# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 110.0)  # 亮度 130
while True:
    print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
    print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
    print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
    print('色调:', cap.get(cv2.CAP_PROP_HUE))
    print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))
    ret, cv_img = cap.read()
    cv2.imshow("Image", cv_img)
    cv2.waitKey(1)
    # cap.release()