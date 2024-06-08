import numpy as np
import cv2 as cv
import time

# 图像旋转函数
def ImageRotate(img,angle):
    # img 输入图像 image_rotation 输出图像 angle 旋转角度
    height,width = img.shape[:2] # (H,W,C) C 图像通道数
    center = (width // 2,height // 2) # 绕图片中心进行旋转
    img_rotation = cv.getRotationMatrix2D(center,angle,1.0) # 逆时针旋转
    return img_rotation

# 取圆形ROI区域函数：输入原图，取原图最大可能的圆形区域输出
def circle_tr(src):
    dst = np.zeros(src.shape,np.uint8) # 感兴趣区域ROI
    mask = np.zeros(src.shape,dtype="uint8") # 感兴趣区域ROI
    (h,w) = mask.shape[:2]
    (cX,cY) = (w // 2,h // 2) # 向下取整
    radius= int(min(h,w) / 2)
    cv.circle(mask,(cX,cY),radius,(255,255,255),-1) # 白色
    # 遍历灰度图的每行每列
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row,col] != 0:
                dst[row,col] = src[row,col]
            elif mask[row,col] == 0:
                dst[row,col] = 0
    return dst

# 金字塔下采样,减小图像分辨率，提高匹配速度
# 每一层是下一层的1/4，上采样是放大图像，下采样是缩小图像
def ImagePyrDown(image,NumLevels):
    for i in range(NumLevels):
        image = cv.pyrDown(image)
    return image

# 旋转匹配函数
# modelPicture 模板图像 searchPicture 待匹配图像
def RotationMatch(modelPicture,searchPicture):

    # 缩小图像
    searchtmp = ImagePyrDown(searchPicture,3)
    modeltmp = ImagePyrDown(modelPicture,3)

    # 得到最大的圆形区域
    newIm = circle_tr(modeltmp)
    # 使用matchTemplate匹配原始灰度图像和模板
    # res = cv.matchTemplate(searchtmp,newIm,cv.TM_SQDIFF_NORMED)
    res = cv.matchTemplate(searchtmp, newIm, cv.TM_CCOEFF)
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
    # location = min_loc
    location = max_loc
    # temp = min_val
    temp = max_val
    angle = 0 # 当前旋转角度为0

    tic = time.time()
    # 以步长为5进行第一次粗循环匹配
    for i in range(-180,181,5):
        newIm = ImageRotate(modeltmp,i)
        newIm = circle_tr(newIm)
        # res = cv.matchTemplate(searchtmp,newIm,cv.TM_SQDIFF_NORMED)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_CCOEFF)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
        # if min_val < temp:
        if max_val > temp:
            # temp = min_val
            temp = max_val
            # location = min_loc
            location = max_loc
            angle = i
    toc = time.time() # 单位是秒
    print('第一次粗循环匹配所花时间为：'+ str(1000*(toc-tic))+'ms')

    tic = time.time()
    # 在当前最优匹配角度周围10的区间以步长为1进行循环匹配计算
    for j in range(angle-5, angle+6):
        newIm = ImageRotate(modeltmp, j)
        newIm = circle_tr(newIm)
        # res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # if min_val < temp:
        if max_val > temp:
            # temp = min_val
            temp = max_val
            # location = min_loc
            location = max_loc
            angle = j
    toc = time.time()  # 单位是秒
    print('在当前最优匹配角度周围10的区间以步长为1进行循环匹配计算所花时间为：' + str(1000 * (toc - tic)) + 'ms')

    tic = time.time()
    # 在当前最优匹配角度周围2的区间以步长为0.1进行循环匹配计算
    k_angle = angle - 0.9
    for k in range(0,19):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(modeltmp, k)
        newIm = circle_tr(newIm)
        # res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # if min_val < temp:
        if max_val > temp:
            # temp = min_val
            temp = max_val
            # location = min_loc
            location = max_loc
            angle = k_angle
    toc = time.time()  # 单位是秒
    print('在当前最优匹配角度周围2的区间以步长为0.1进行循环匹配计算所花时间为：' + str(1000 * (toc - tic)) + 'ms')

    # 用下采样前的图像进行精匹配计算
    k_angle = angle - 0.1
    newIm = ImageRotate(modelPicture,k_angle)
    newIm = circle_tr(newIm)
    # res = cv.matchTemplate(searchPicture,newIm,cv.TM_CCORR_NORMED)
    res = cv.matchTemplate(searchPicture, newIm, cv.TM_CCORR)
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
    location = max_loc
    temp = max_val
    angle = k_angle
    result = res
    modelIm = newIm
    for k in range(1,3):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(modelPicture,k_angle)
        newIm = circle_tr(newIm)
        # res = cv.matchTemplate(searchPicture, newIm, cv.TM_CCORR_NORMED)
        res = cv.matchTemplate(searchPicture, newIm, cv.TM_CCORR)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val > temp:
            temp = max_val
            location = max_loc
            angle = k_angle
            result = res
            modelIm = newIm

    # 多目标匹配
    # tic = time.time()
    # threshold = 0.995
    # 第一次筛选
    # loc = np.where(result >= threshold)
    # 第二次筛选：将位置偏移小于5个像素的结果舍去
    # for other_loc in zip(*loc[::-1]):
    #    if (location[0] + 5 < other_loc[0]) or (location[1] + 5 < other_loc[1]):
    #        location = other_loc
            # color (blue,green,red)
    #        cv.rectangle(searchPicture, other_loc, (other_loc[0] + modelPicture.shape[1], other_loc[1] + modelPicture.shape[0]), (0, 0, 255), 2)
    # toc = time.time()
    # print('多目标匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    print('最好匹配指数'+str(temp))
    # cv.namedWindow("multi result", cv.WINDOW_NORMAL)
    # cv.imshow("multi result", searchPicture)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 为什么需要加50
    # location_x = location[0] + 50
    # location_y = location[1] + 50
    location_x = location[0]
    location_y = location[1]

    # 待检测图像应该旋转的角度
    angle = -angle
    match_point = {'angle':angle,'point':(location_x,location_y)}
    return match_point

# 画图
def draw_result(src,temp,match_point):
    cv.rectangle(src,match_point,(match_point[0]+temp.shape[1],match_point[1]+temp.shape[0]),(0,255,0),2)
    cv.namedWindow("result", cv.WINDOW_NORMAL)
    cv.imshow("result", src)

    cv.waitKey(0)
    cv.destroyAllWindows()

# 预处理输入图像
def get_realsense(src,temp):
    SearchImage = src
    ModelImage = temp
    # 高斯滤波：减少噪声;Canny边缘检测
    ModelImage_edge = cv.GaussianBlur(ModelImage, (5, 5), 0)
    ModelImage_edge = cv.Canny(ModelImage_edge, 10, 200, apertureSize=3)
    SearchImage_edge = cv.GaussianBlur(SearchImage, (5, 5), 0)

    (h1, w1) = SearchImage_edge.shape[:2]
    SearchImage_edge = cv.Canny(SearchImage_edge, 10, 180, apertureSize=3)
    # search_ROIPart = SearchImage_edge[50:h1 - 50, 50:w1 - 50]  # 裁剪图像
    search_ROIPart = SearchImage_edge

    tic = time.time()
    match_points = RotationMatch(ModelImage_edge, search_ROIPart)
    # match_points = RotationMatch(ModelImage, SearchImage)
    toc = time.time()
    print('匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    print('匹配的最优区域的起点坐标为：' + str(match_points['point']))
    print('相对旋转角度为：' + str(match_points['angle']))

    draw_result(SearchImage, ModelImage_edge, match_points['point'])

if __name__ == '__main__':
    # 模板匹配 多角度
    # 单一匹配 效果很差；从原图抠图效果也差  多目标匹配 从原图抠图效果很差；反模式的stronglyLostData效果还行
    # path1 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_AllErrors.jpg"
    path1 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_AllErrors_NoText.jpg"

    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/MissingData.jpg" # 0.934 匹配歪了
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/RedundantData.jpg" # 0.934 匹配歪了
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/StronglyLostData.jpg" # 0.995 匹配到一半 1.00+ no text
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/InconsistentData.jpg" # 0.934 匹配歪了 0.936 no text

    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_MissingData.jpg"
    path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_RedundantData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_StronglyLostData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_InconsistentData.jpg"
    src = cv.imread(path1,0)
    temp = cv.imread(path2,0)
    get_realsense(src,temp)

    # 匹配方式换成cv2.TM_CCOEFF效果也很差；不应该使用旋转