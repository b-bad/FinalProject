import os
import cv2
import numpy as np
from scipy import stats
import argparse

"""parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input', type=str, help="Path to the input image")
args = vars(parser.parse_args())
IMAGE_PATH = args["input"]"""


class AngleMeasure:

    def __init__(self, IMAGE_PATH):

        self.isInputExist = False
        self.outputDirPath = './output'
        self.inputPath = IMAGE_PATH
        if os.path.exists(self.inputPath):
            self.img = cv2.imread(self.inputPath)
            self.gray = cv2.GaussianBlur(cv2.imread(self.inputPath, 0),
                                         (5, 5), 0)
            self.kernel = np.ones((3, 3), np.uint8)
            self.dilation = cv2.dilate(self.gray, self.kernel, iterations=1)
            self.canny = cv2.Canny(self.gray, 150, 250, apertureSize=3)
            self.cannyD = cv2.Canny(self.dilation, 150, 250, apertureSize=3)
            self.threshold = 900
            self.isInputExist = True
            self.flag = False
            self.angleList = None
            self.result = None
        else:
            print("input image is not exist")
            exit()

    def draw_lines(self, input_img, lines, a_l):

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = (x0 + 1000 * (-b))
            y1 = (y0 + 1000 * a)
            x2 = (x0 - 1000 * (-b))
            y2 = (y0 - 1000 * a)
            hough = cv2.line(input_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # print([x1, x2, y1, y2])
            # angle = int(theta / np.pi * 180)
            angle = theta / np.pi * 180
            angle, a_l = self.result_push(a_l, angle)

        return hough, a_l

    def draw_lines_p(self, input_image, lines, a_l):

        for line in lines:
            x1, y1, x2, y2 = line[0]
            hough = cv2.line(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x2 == x1:
                angle = 90
            else:
                # angle = int(np.arctan((y2 - y1) / (x2 - x1)) * 180)
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180
            angle, a_l = self.result_push(a_l, angle, axis=1)

        return hough, a_l

    def result_push(self, a_l, angle, axis=0):
        if axis:
            if angle >= 360:
                angle -= 360
            elif angle >= 180:
                angle -= 180
            elif angle <= -360:
                angle = -angle - 360
            elif angle <= -180:
                angle = -angle - 180
            elif angle <= 0:
                angle = -angle
        else:
            if 0 <= angle <= 90:
                angle = 90 - angle
            elif 90 < angle <= 270:
                angle = 270 - angle
            elif 270 < angle <= 360:
                angle = 450 - angle
        a_l.append(angle)

        return angle, a_l

    def result_process(self, a_l):
        for a in a_l:
            if a >= 90:
                a -= 90
        a_l = np.array(a_l)
        a_l_unique = np.unique(a_l)
        a_l_copy = []
        if len(a_l_unique) == 1:
            result = a_l[0]
        else:
            mode = stats.mode(a_l)[0][0]
            for i in a_l:
                if mode - 10 <= i <= mode + 10:
                    a_l_copy.append(i)
            result = np.mean(a_l_copy)

        return result

    def measure(self):

        y, x, z = self.img.shape
        self.angleList = []

        while self.threshold > 0 and self.flag is False:

            lines = cv2.HoughLines(self.canny, 1.0, np.pi / 180, self.threshold)
            lines_ = []
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = (x0 + 1000 * (-b))
                    y1 = (y0 + 1000 * a)
                    x2 = (x0 - 1000 * (-b))
                    y2 = (y0 - 1000 * a)
                    if not (((x1<15 and x2<15) or (x1 > (x-15) and x2 > (x-15))) or ((y1<15 and y2<15) or (y1>y-15 and y2>y-15))):
                        lines_.append(line)
                        # print(self.threshold)
                if lines_.__len__():
                    hough, self.angleList = self.draw_lines(self.img.copy(), lines_, self.angleList)
                    self.flag = True
                else:
                    self.threshold -= 100

            else:
                self.threshold -= 100

        if self.flag is False:

            lines_with_p = cv2.HoughLinesP(self.cannyD, 1.0, np.pi / 180, 100, max(x, y) / 10, 10)

            if lines_with_p is not None:
                hough, self.angleList = self.draw_lines_p(self.img.copy(), lines_with_p, self.angleList)

            else:
                hough = self.img.copy()
        cv2.imshow("h", cv2.resize(hough, (x//3, y//3)))
        cv2.waitKey(0)
        self.result = self.result_process(self.angleList)
        # print(self.inputPath + "result: " + str(int(self.result)))

        if 45 <= abs(self.result) < 135:
            if self.result > 0:
                self.result -= 90
            else:
                self.result += 90
        elif 135 <= abs(self.result) <180:
            if self.result > 0:
                self.result -= 180
            else:
                self.result += 180

        # print(self.angleList)
        # print(self.inputPath + "result: " + str(self.result))

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # print("RotationMatrix2D：\n", M)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    print("RotationMatrix2D：\n", M)

    # 执行仿射变换、得到图像
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    # borderMode=cv2.BORDER_REPLICATE 使用边缘值填充
    # 或使用borderValue=(255,255,255)) # 使用常数填充边界（0,0,0）表示黑色

if __name__ == "__main__":
    path = r"C:\Users\Administrator_wzz\Desktop\0218_raw_r"
    file_list = os.listdir(path)
    file_list_ = [path + "/" + file for file in file_list]
    for file in file_list_:
        am = AngleMeasure(file)
        print(file)
        am.measure()
        angle_result = am.result
        img_r = rotate_bound(am.img, -am.result)
        cv2.imwrite(file + "_r.jpg", img_r)