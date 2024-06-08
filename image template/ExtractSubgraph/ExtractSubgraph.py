import numpy as np
import cv2 as cv

def extractSubgraph_SIFT(path1,path2):
    # 读入图像
    img_name_1 = path1
    img_name_2 = path2

    img_1 = cv.imread(img_name_1)
    img_2 = cv.imread(img_name_2)

    # 转换为灰度图像
    gray_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    gray_2 = cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)

    # 实例化SIFT算子
    sift = cv.SIFT_create()

    # 分别对两张图像进行SIFT检测
    kp_1,des_1 = sift.detectAndCompute(img_1,None)
    kp_2,des_2 = sift.detectAndCompute(img_2,None)

    # 显示特征点
    img_res_1 = cv.drawKeypoints(img_1,kp_1,gray_1,color=(255,0,255))
    img_res_2 = cv.drawKeypoints(img_2,kp_2,gray_2,color=(0,0,255))
    cv.imshow("SIFT_image_1",img_res_1)
    cv.imshow("SIFT_image_2",img_res_2)

    # BFMatcher算法匹配
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_1,des_2,k=2)

    # 筛选优质的匹配点
    ratio = 0.99
    good_features = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_features.append([m])

    # 将匹配的特征点绘制在一张图内
    img_res = cv.drawMatchesKnn(img_1,kp_1,img_2,kp_2,good_features,None,flags=2)
    cv.namedWindow("BFmatch", cv.WINDOW_NORMAL)
    cv.imshow("BFmatch",img_res)

    cv.waitKey(0)
    cv.destroyAllWindows()

def extractSubgraph_matchTemplate(path1,path2):
    img = cv.imread(path1,0)
    template = cv.imread(path2,0)
    w,h = template.shape[::-1]

    # 匹配
    method = cv.TM_CCORR
    res = cv.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w,top_left[1] + h)

    # 在原始图像上框出匹配图像
    cv.rectangle(img,top_left,bottom_right,255,2)
    text = 'Detected Point'
    cv.namedWindow(text, cv.WINDOW_NORMAL)
    cv.imshow(text, img)
    cv.waitKey()
    cv.destroyAllWindows()

def extractSubgraph_matchTemplateMulti(path1,path2):
    # 读取目标图片和模板图片
    target = cv.imread(path1)
    template = cv.imread(path2)

    th,tw = template.shape[:2]
    result = cv.matchTemplate(target,template,cv.TM_SQDIFF_NORMED)
    print('result:',result)
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)

    # 绘制矩形边框，将匹配区域标注出来
    cv.rectangle(target,min_loc,(min_loc[0]+tw,min_loc[1]+th),(0,0,255),2)
    strmin_val = str(min_val)
    temp_loc = min_loc
    numOfloc = 1

    # 设置匹配阈值为0.01/0.005
    threshold = 0.005
    # 第一次筛选
    loc = np.where(result < threshold)
    # 第二次筛选：将位置偏移小于5个像素的结果舍去
    for other_loc in zip(*loc[::-1]):
        if (temp_loc[0]+5 < other_loc[0]) or (temp_loc[1]+5<other_loc[1]):
            numOfloc = numOfloc + 1
            temp_loc = other_loc
            cv.rectangle(target,other_loc,(other_loc[0]+tw,other_loc[1]+th),(0,0,225),2)
    str_numOfloc = str(numOfloc)

    strText = "MatchResult--MatchingValue="+strmin_val+"--NumberOfPosition"+str_numOfloc
    cv.namedWindow(strText,cv.WINDOW_NORMAL)
    cv.imshow(strText,target)
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':

    # path1 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_AllErrors.jpg"
    path1 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_AllErrors_NoText.jpg"

    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/MissingData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/RedundantData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/StronglyLostData.jpg" # 0.005
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/AntiPatterns/InconsistentData.jpg" # 0.005

    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_MissingData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_RedundantData.jpg"
    path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_StronglyLostData.jpg"
    # path2 = "D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_InconsistentData.jpg"

    # 不管原始图形去不去掉文字，和单独的反模式匹配效果都很差
    # 特征点提取：SIFT  反模式的每个图形拆开匹配，全图的矩形和椭圆都匹配上了；从原图抠反模式也是如此
    # extractSubgraph_SIFT(path1,path2)

    # 模板匹配 matchTemplate 单一匹配 匹配效果连边都不沾；从原图抠反模式也是如此
    # extractSubgraph_matchTemplate(path1,path2)

    # 模板匹配 matchTemplate 多目标匹配 跟反模式匹配效果不好；从原图抠图可以检测出来
    extractSubgraph_matchTemplateMulti(path1,path2)