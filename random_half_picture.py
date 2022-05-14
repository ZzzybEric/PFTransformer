import copy
import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate=0.1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        #sample2 =copy.copy(sample)
        #shutil.copyfile(sample2,"/home/zhaoyunbin/数据集/partial_dataset/OccludedREID/train")
        for name in sample:
                shutil.move(fileDir+name, "/home/zhaoyunbin/数据集/Market/trainselect/"+name)
        return

if __name__ == '__main__':
    fileDir = "/home/zhaoyunbin/数据集/Market/bounding_box_train/"    #源图片文件夹路径
    tarDir = './result/'    #移动到新的文件夹路径
    moveFile(fileDir)
