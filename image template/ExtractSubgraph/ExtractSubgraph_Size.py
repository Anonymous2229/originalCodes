import numpy as np
import argparse # 把命令行输入的命令转换成python数据
import imutils
import glob # 查找文件目录和文件，将搜索结果返回到列表中
import cv2
import time

def extractSubgraph_Size():

    ap = argparse.ArgumentParser()
    # 模板图像和待匹配图像的路径
    ap.add_argument("-t","--template",required=True,help="Path to template image")
    ap.add_argument("-i","--image",required=True,help="Path to image where template will be matched")
    # 每次循环是否可视
    ap.add_argument("-v","--visualize",help="Flag indicating whether or not to visualize each iteration")
    args = vars(ap.parse_args())

    # 原始图像变成灰色图像，进行边缘处理
    tic = time.time()
    image = cv2.imread(args["image"])
    newImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    newImage = cv2.Canny(newImage,50,200)
    # 原始图像的宽高
    ih,iw = newImage.shape[:2]

    # 读取模板图像
    template = cv2.imread(args["template"])
    # 灰色图像处理
    gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    found = None
    # 模板图像循环缩小   起始值，结束值，相等的块片数
    for scale in np.linspace(0.2,1.0,20)[::-1]:
        resized = imutils.resize(gray,width=int(gray.shape[1]*scale))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] >= ih or resized.shape[1] >= iw:
            continue

        # 模板图像边缘处理
        edged = cv2.Canny(resized,50,200)
        # result = cv2.matchTemplate(newImage,edged,cv2.TM_CCOEFF)
        result = cv2.matchTemplate(newImage, edged, cv2.TM_CCOEFF_NORMED) # 匹配原图抠图效果很好
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)

        # 检查每次循环是否可视化
        if args.get("visualize",False):
            cv2.rectangle(image,(max_loc[0],max_loc[1]),(max_loc[0]+edged.shape[1],max_loc[1]+edged.shape[0]),(0,0,255),2)
            cv2.namedWindow("Visualize", cv2.WINDOW_NORMAL)
            cv2.imshow("Visualize",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if found is None or max_val > found[0]:
            found = (max_val,max_loc,result,r)

    (max_val,max_loc,result,r) = found
    print(max_val)

    # 找到最佳匹配位置
    startX,startY = max_loc[0],max_loc[1]
    endX,endY = int(max_loc[0]+r*template.shape[1]),int(max_loc[1]+r*template.shape[0])
    toc = time.time()
    print('多尺度模板匹配花费时间：'+str((toc-tic)*1000)+'ms')

    # 画图
    cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
    cv2.namedWindow("SearchImage", cv2.WINDOW_NORMAL)
    cv2.imshow("SearchImage",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    extractSubgraph_Size()
    # Terminal命令
    # python ExtractSubgraph_Size.py --template D:/PyCharm/ExtractSubgraph/resources/PaperReview/antiPatterns/MissingData.jpg --image D:/PyCharm/ExtractSubgraph/resources/PaperReview/composeMistakes/PaperReview_StronglyLost.jpg
    # 反模式 stronglyLost/inconsistent 不准
    # missing 范围圈大 999.6256828308105ms redundant 范围圈大 992.02561378479ms weaklyLost 范围圈大 1171.4046001434326ms
    # python ExtractSubgraph_Size.py --template D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_MissingData.jpg --image D:/PyCharm/ExtractSubgraph/resources/PaperReview/PaperReview_For_AllErrors.jpg
    # 原图抠图 missing 873.2783794403076ms redundant 792.1230792999268ms inconsistent 973.24538230896ms stronglyLost 925.180196762085ms

    # 归一化方法的匹配效果没有不归一好，取低值的最佳匹配没有取高值的最佳匹配效果好
    # 综合选择 TM_CCORR 或 TM_CCOEFF
    # 选择这两种匹配方式就不能用阈值限制了，可能无限大（？）；那怎么匹配多个错误呢？

    # 归一化方法适合匹配原图抠子图；但实际情况不可能从原图抠图

    # BookingSystem 反模式从发生数据流错误的地方去文字抠图，粘贴到别的地方
    # python ExtractSubgraph_Size.py --template D:/PyCharm/ExtractSubgraph/resources/BookingSystem/BookingSystem_MissingData.png --image D:/PyCharm/ExtractSubgraph/resources/BookingSystem/BookingSystem_For_AllErrors.png
    # missing 1199.0396976470947ms redundant 1152.8277397155762ms stronglyLost 不准
    # PastryCook 只能测一个，反模式得从发生数据流错误的地方去文字抠图然后移到别的地方
    #  python ExtractSubgraph_Size.py --template D:/PyCharm/ExtractSubgraph/resources/PastryCook/PastryCook_MissingData.png --image D:/PyCharm/ExtractSubgraph/resources/PastryCook/PastryCook_For_AllErrors.png
    # missing 1187.192440032959ms redundant 1214.2863273620605ms






