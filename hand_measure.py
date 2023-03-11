# https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
import cv2
import mediapipe as mp
import time
import math

# cap = cv2.VideoCapture(0)
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
        ((hand_[0][0] - hand_[2][0]), (hand_[0][1] - hand_[2][1]), (hand_[0][2] - hand_[2][2])),
        ((hand_[3][0] - hand_[4][0]), (hand_[3][1] - hand_[4][1]), (hand_[3][2] - hand_[4][2])))
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[5][0])), (int(hand_[0][1]) - int(hand_[5][1])),
         (int(hand_[0][2]) - int(hand_[5][2]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])),
         (int(hand_[7][2]) - int(hand_[8][2])))

    )

    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[9][0])), (int(hand_[0][1]) - int(hand_[9][1])),
         (int(hand_[0][2]) - int(hand_[9][2]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])),
         (int(hand_[11][2]) - int(hand_[12][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 無名指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[13][0])), (int(hand_[0][1]) - int(hand_[13][1])),
         (int(hand_[0][2]) - int(hand_[13][2]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])),
         (int(hand_[15][2]) - int(hand_[16][2])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle_3d(
        ((int(hand_[0][0]) - int(hand_[17][0])), (int(hand_[0][1]) - int(hand_[17][1])),
         (int(hand_[0][2]) - int(hand_[17][2]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])),
         (int(hand_[19][2]) - int(hand_[20][2])))
    )

    angle_list.append(angle_)

    return angle_list


# cap = cv2.VideoCapture()
# # The device number might be 0 or 1 depending on the device and the webcam
# cap.open(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 80.0)  # 亮度 130
# cap.set(cv2.CAP_PROP_CONTRAST, 128.0)  # 對比度 32
# cap.set(cv2.CAP_PROP_SATURATION, 128.0)  # 飽和度 64
# cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色調 0
# cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)  # 曝光 -4
# while True:
def hand_input(img):
    # success, img = img.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    angle_list = [0, 0, 0, 0, 0]
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
                # print(angle_list

            angle_3d = []
            for i in range(21):
                x = handLms.landmark[i].x * 100000
                y = handLms.landmark[i].y * 100000
                z = handLms.landmark[i].z * 100000
                # print([x, y, z])
                angle_3d.append([x, y, z])
            if angle_3d != []:
                angle_list = hand_angle_3d(angle_3d)
                # print(angle_list)
                # cv2.putText(img, str(angle_list), (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cTime = time.time()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return angle_list, img


def main():
    # from os import listdir
    # from os.path import isfile, join
    # mypath = 'C:\\Users\\User\\PycharmProjects\\test_msi\\save_img\\2023_03_01_save_img'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for item in onlyfiles:
    #     print(item)
    from os import listdir
    from os.path import isfile, join
    import matplotlib.pyplot as plt
    dir_list = [
        'C:\\Users\\User\\PycharmProjects\\test_msi\\save_img\\2023_03_01_save_img_3',
        # 'C:\\Users\\User\\PycharmProjects\\test_msi\\save_img\\2023_03_01_save_img_2',
        # 'C:\\Users\\User\\PycharmProjects\\test_msi\\save_img\\2023_03_01_save_img',
        # 'C:\\Users\\User\\PycharmProjects\\test_msi\\save_img\\2023_03_01_save_img_4'
    ]

    for mypath in dir_list:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        # print(len(onlyfiles))
        all_hand_angle = {}
        thumb_list = []
        index_list = []
        ring_list = []
        middle_list = []
        pink_list = []

        for item in onlyfiles:
            # print(item)
            img = cv2.imread(mypath + '\\' + item)
            angle_list, img_ = hand_input(img)
            hand_ = item.split('.')[0] + 'hand_detection'
            print(mypath + '\\' + hand_)
            status = cv2.imwrite(mypath + '\\' + hand_ + '.PNG', img_)
            all_hand_angle.update({item: angle_list})
            thumb_list.append(angle_list[0])
            index_list.append(angle_list[1])
            ring_list.append(angle_list[2])
            middle_list.append(angle_list[3])
            pink_list.append(angle_list[4])
        print(max(thumb_list), max(index_list), max(ring_list), max(middle_list), max(pink_list))
        print(min(thumb_list), min(index_list), min(ring_list), min(middle_list), min(pink_list))
        y1 = thumb_list
        x1 = list(range(1, len(onlyfiles) + 1))
        # 繪製折線圖，顏色「紅色」，線條樣式「-」，線條寬度「2」，標記大小「16」，標記樣式「.」，圖例名稱「Plot 1」
        plt.xlim(0, 50)  # 設定 x 軸座標範圍
        plt.ylim(0, 180)  # 設定 y 軸座標範圍
        plt.xlabel('image_number', fontsize="10")  # 設定 x 軸標題內容及大小
        plt.ylabel('angle', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('thumb_list', fontsize="18")  # 設定圖表標題內容及大小
        plt.plot(x1, y1, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="Plot 1")
        plt.savefig(mypath + '\\' + 'thumb_list' + '.PNG')
        plt.show()
        # plt.gcf().savefig('thumb_list.jpg'
        # status = cv2.imwrite(mypath + '\\' + 'thumb_list' + '.PNG', plt)

        y1 = index_list
        # 繪製折線圖，顏色「紅色」，線條樣式「-」，線條寬度「2」，標記大小「16」，標記樣式「.」，圖例名稱「Plot 1」
        plt.xlim(0, 50)  # 設定 x 軸座標範圍
        plt.ylim(0, 180)  # 設定 y 軸座標範圍
        plt.xlabel('image_number', fontsize="10")  # 設定 x 軸標題內容及大小
        plt.ylabel('angle', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('index_list', fontsize="18")  # 設定圖表標題內容及大小
        plt.plot(x1, y1, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="Plot 1")
        plt.savefig(mypath + '\\' + 'index_list' + '.PNG')
        plt.show()

        y1 = ring_list
        # 繪製折線圖，顏色「紅色」，線條樣式「-」，線條寬度「2」，標記大小「16」，標記樣式「.」，圖例名稱「Plot 1」
        plt.xlim(0, 50)  # 設定 x 軸座標範圍
        plt.ylim(0, 180)  # 設定 y 軸座標範圍
        plt.xlabel('image_number', fontsize="10")  # 設定 x 軸標題內容及大小
        plt.ylabel('angle', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('ring_list', fontsize="18")  # 設定圖表標題內容及大小
        plt.plot(x1, y1, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="Plot 1")
        plt.savefig(mypath + '\\' + 'ring_list' + '.PNG')
        plt.show()

        y1 = middle_list
        # 繪製折線圖，顏色「紅色」，線條樣式「-」，線條寬度「2」，標記大小「16」，標記樣式「.」，圖例名稱「Plot 1」
        plt.xlim(0, 50)  # 設定 x 軸座標範圍
        plt.ylim(0, 180)  # 設定 y 軸座標範圍
        plt.xlabel('image_number', fontsize="10")  # 設定 x 軸標題內容及大小
        plt.ylabel('angle', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('middle_list', fontsize="18")  # 設定圖表標題內容及大小
        plt.plot(x1, y1, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="Plot 1")
        plt.savefig(mypath + '\\' + 'middle_list' + '.PNG')
        plt.show()

        y1 = pink_list
        # 繪製折線圖，顏色「紅色」，線條樣式「-」，線條寬度「2」，標記大小「16」，標記樣式「.」，圖例名稱「Plot 1」
        plt.xlim(0, 50)  # 設定 x 軸座標範圍
        plt.ylim(0, 180)  # 設定 y 軸座標範圍
        plt.xlabel('image_number', fontsize="10")  # 設定 x 軸標題內容及大小
        plt.ylabel('angle', fontsize="10")  # 設定 y 軸標題內容及大小
        plt.title('pink_list', fontsize="18")  # 設定圖表標題內容及大小
        plt.plot(x1, y1, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label="Plot 1")
        plt.savefig(mypath + '\\' + 'pink_list' + '.PNG')
        plt.show()


main()
