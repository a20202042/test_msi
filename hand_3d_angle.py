# https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
keypoint_pos = []


def angel_check(angel_data):
    hand_angel = [[1, 87], [0, 153], [82, 165], [53, 168], [42, 176]]
    if int(angel_data[0]) in range(hand_angel[0][0], hand_angel[0][1]) and \
            int(angel_data[1]) in range(hand_angel[1][0], hand_angel[1][1]) and \
            int(angel_data[2]) in range(hand_angel[2][0], hand_angel[2][1]) and \
            int(angel_data[3]) in range(hand_angel[3][0], hand_angel[3][1]) and \
            int(angel_data[4]) in range(hand_angel[4][0], hand_angel[4][1]):
        re = 'caliper'
    else:
        re = ''
    return re


def vector_2d_angle(v1, v2):  # 求出v1,v2兩條向量的夾角
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 100000.
    return angle_


def hand_angle(hand_):
    angle_list = []
    # ---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list

def vector_2d_angle_3d(v1, v2):  # 求出v1,v2兩條向量的夾角
    v1_x = v1[0]
    v1_y = v1[1]
    v1_z = v1[2]
    v2_x = v2[0]
    v2_y = v2[1]
    v2_z = v2[2]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y + v1_z * v2_z) / (
                        ((v1_x ** 2 + v1_y ** 2 + v1_z ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2 + v2_z ** 2) ** 0.5))))
    except:
        angle_ = 100000.
    return angle_


def hand_angle_3d(hand_):
    angle_list = []
    # ---------------------------- thumb 大拇指角度
    # ---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1])), (int(hand_[0][2]) - int(hand_[2][2]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])), (int(hand_[3][2]) - int(hand_[4][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1])), (int(hand_[0][2]) - int(hand_[6][2]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])), (int(hand_[7][2]) - int(hand_[8][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1])), (int(hand_[0][2]) - int(hand_[10][2]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])), (int(hand_[11][2]) - int(hand_[12][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 無名指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1])), (int(hand_[0][2]) - int(hand_[14][2]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])), (int(hand_[15][2]) - int(hand_[16][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1])), (int(hand_[0][2]) - int(hand_[18][2]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])), (int(hand_[19][2]) - int(hand_[20][2])))
    )

    angle_list.append(angle_)

    return angle_list

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        keypoint_pos = []
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id ==0:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for i in range(21):
                x = handLms.landmark[i].x * imgRGB.shape[1]
                y = handLms.landmark[i].y * imgRGB.shape[0]
                keypoint_pos.append((x, y))
            if keypoint_pos:
                # 得到各手指的夾角資訊
                angle_list = hand_angle(keypoint_pos)
                ca = angel_check(angel_data=angle_list)
                print(angle_list, ',')
                # if ca == 'caliper':
                #     # print('caliper')
                #     cv2.putText(img, str('caliper'), (10, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                # else:
                #     cv2.putText(img, str('no_caliper'), (10, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            angle_3d = []
            for i in range(21):
                x = handLms.landmark[i].x * 100000
                y = handLms.landmark[i].y * 100000
                z = handLms.landmark[i].z * 100000
                print([x, y, z])
                angle_3d.append([x, y, z])
            if angle_3d != []:
                angle_list = hand_angle_3d(angle_3d)
                print(angle_list)
                cv2.putText(img, str(angle_list), (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
