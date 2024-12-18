import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch

from AlgorithmPool.FunctionalMethodPool.DataIO import NIIread
from AlgorithmPool.FunctionalMethodPool.DataIO import commonFunc
from AlgorithmPool.FunctionalMethodPool.DataIO import writeNII
from AlgorithmPool.NetPool.EvalFunction import IOU


# 定义读取文件的格式
def default_loader(path):
    return NIIread.NiiRead(path)


def default_loader_2Dnii(path):
    return NIIread.NiiRead2D(path)


# 定义加载动脉瘤数据方式
class MyDataset(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImg.nii', '_label.nii',
                                                                            '_headMask.nii')
        datas = []
        for (i, j, k) in zip(dataList, labelList, addData1):
            datas.append((i, j, k))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, vessel = self.datas[index]
        img = self.loader(data)
        imgT = img[0].copy().astype(np.float32)
        vesselImg = self.loader(vessel)
        vesselT = vesselImg[0].copy().astype(np.float32)
        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT > 0)
        if self.transform is not None:
            imgT = self.transform(np.multiply(imgT, vesselT))  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader_2Dnii):
        super(MyDatasetFYY, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImg.nii', '_label.nii')
        datas = []
        for (i, j, k) in zip(dataList, labelList, addData1):
            datas.append((i, j, k))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, vessel = self.datas[index]
        img = self.loader(data)
        imgT = img[0].copy().astype(np.float32)
        # imgT = np.multiply((-1 * imgT) + max(imgT.reshape(-1)), imgT > 0)
        # writeNII.writeArrayToNii(torch.from_numpy(imgT), r'E:\data\brainLesion\MR_2D/', 'imgT.nii')
        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        # imgTagT = np.multiply(imgTagT, imgT > 0)
        # writeNII.writeArrayToNii(torch.from_numpy(imgTagT), r'E:\data\1\MR_2D/', 'imgTagT.nii')
        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY3D(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY3D, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImg.nii', '_label.nii')
        datas = []
        for (i, j, k) in zip(dataList, labelList, addData1):
            datas.append((i, j, k))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, vessel = self.datas[index]
        img = self.loader(data)
        imgT = img[0].copy().astype(np.float32)
        # imgT = np.multiply((-1 * imgT) + max(imgT.reshape(-1)), imgT > 0)
        # writeNII.writeArrayToNii(torch.from_numpy(imgT), r'E:\data\brainLesion\MR_2D/', 'imgT.nii')
        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        # imgTagT = np.multiply(imgTagT, imgT > 0)
        # writeNII.writeArrayToNii(torch.from_numpy(imgTagT), r'E:\data\1\MR_2D/', 'imgTagT.nii')
        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_DWI(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_DWI, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImgADC.nii', '_label.nii',
                                                                            '_orgImgB0.nii', '_orgImgB1000.nii')
        datas = []
        for (i, j, k, l) in zip(dataList, labelList, addData1, addData2):
            datas.append((i, j, k, l))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, datab0, datab1000 = self.datas[index]
        img0 = self.loader(data)
        imgT0 = img0[0].copy().astype(np.float32)

        img1 = self.loader(datab0)
        imgT1 = img1[0].copy().astype(np.float32)

        img2 = self.loader(datab1000)
        imgT2 = img2[0].copy().astype(np.float32)
        # 将data和datab1000进行通道合并,最多传3D传不了4D
        # imgT0 = imgT0[np.newaxis, :]
        # imgT1 = imgT1[np.newaxis, :]
        # imgT2 = imgT2[np.newaxis, :]
        imgT = np.concatenate((imgT0, imgT1, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT0 > 0)
        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_ISLES(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_ISLES, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3 = commonFunc.get_SpliteList3(path, '_orgImgDWI.nii',
                                                                                       '_label.nii',
                                                                                       '_orgImgFlair.nii',
                                                                                       '_orgImgT1.nii', '_orgImgT2.nii')
        datas = []
        for (i, j, k, l, m) in zip(dataList, labelList, addData1, addData2, addData3):
            datas.append((i, j, k, l, m))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, dataFlair, dataT1, dataT2 = self.datas[index]
        img0 = self.loader(data)
        imgT0 = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFlair)
        imgT1 = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT2 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT3 = img3[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT0, imgT1, imgT2, imgT3), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT0 > 0)
        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_ISLES2022(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_ISLES2022, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImgADC.nii', '_label.nii',
                                                                            '_orgImgB1000.nii')
        datas = []
        for (i, j, k) in zip(dataList, labelList, addData1):
            datas.append((i, j, k))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label, datab1000 = self.datas[index]
        img0 = self.loader(data)
        imgT0 = img0[0].copy().astype(np.float32)

        img1 = self.loader(datab1000)
        imgT1 = img1[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT0, imgT1), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor

        # savePath = r"D:\data/"
        # i = 0
        # writeNII.writeArrayToNii(imgT[ 0:80, :, :], savePath, str(i) + 'imgT0' + '.nii')
        # writeNII.writeArrayToNii(imgT[ 80:160, :, :], savePath, str(i) + 'imgT1' + '.nii')
        # writeNII.writeArrayToNii(imgTagT, savePath, str(i) + 'imgTagT' + '.nii')

        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_roi.nii', '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_pyG.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)

        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT1ce > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
            imgGT = self.transform(imgGT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_Txt(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, txt_name, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_Txt, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList_txt(path, txt_name,
                                                                                                    '_T1ce.nii',
                                                                                                    '_roi.nii',
                                                                                                    '_FL.nii',
                                                                                                    '_T1.nii',
                                                                                                    '_T2.nii',
                                                                                                    '_skullMask.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)

        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        # imgTagT = np.multiply(imgTagT, imgT1ce > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
            imgGT = self.transform(imgGT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_Txt_skull(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, txt_name, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_Txt_skull, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList_txt(path, txt_name,
                                                                                                    '_T1ce.nii',
                                                                                                    '_roi.nii',
                                                                                                    '_FL.nii',
                                                                                                    '_T1.nii',
                                                                                                    '_T2.nii',
                                                                                                    '_skullMask.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]

        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)
        imgT1ce = np.multiply(imgT1ce, imgGT > 0)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)
        imgFl = np.multiply(imgFl, imgGT > 0)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)
        imgT1 = np.multiply(imgT1, imgGT > 0)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)
        imgT2 = np.multiply(imgT2, imgGT > 0)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT1ce > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
            imgGT = self.transform(imgGT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_py(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_py, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_roi.nii', '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_skullMask.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)

        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        # imgTagT = np.multiply(imgTagT, imgT1ce > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
            imgGT = self.transform(imgGT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_skull_old(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_skull_old, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_roi.nii', '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_skullMaskR.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)

        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgT1ce > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
            imgGT = self.transform(imgGT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_skull(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_skull, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_roi.nii', '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_skullMask.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)
        imgT1ce = np.multiply(imgT1ce, imgGT > 0)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)
        imgFl = np.multiply(imgFl, imgGT > 0)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)
        imgT1 = np.multiply(imgT1, imgGT > 0)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)
        imgT2 = np.multiply(imgT2, imgGT > 0)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)
        imgTagT = np.multiply(imgTagT, imgGT > 0)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_skull_t(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_skull_t, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_roi.nii', '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_skullMask.nii')
        # '_skullMask_brats.nii')
        datas = []
        for (i, j, k, l, m, n) in zip(dataList, labelList, addData1, addData2, addData3, addData4):
            datas.append((i, j, k, l, m, n))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, label, dataFl, dataT1, dataT2, dataG = self.datas[index]
        img4 = self.loader(dataG)
        imgGT = img4[0].copy().astype(np.float32)

        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)
        imgT1ce = np.multiply(imgT1ce, imgGT > 0)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)
        imgFl = np.multiply(imgFl, imgGT > 0)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)
        imgT1 = np.multiply(imgT1, imgGT > 0)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)
        imgT2 = np.multiply(imgT2, imgGT > 0)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgGT = self.target_transform(imgGT)  # 数据标签转换为Tensor
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgGT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_TCGA_pre(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_TCGA_pre, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2, addData3, addData4 = commonFunc.get_SpliteList3(path, '_T1ce.nii',
                                                                                                 '_FL.nii',
                                                                                                 '_T1.nii', '_T2.nii',
                                                                                                 '_T2.nii')
        datas = []
        for (i, j, k, l) in zip(dataList, labelList, addData1, addData2):
            datas.append((i, j, k, l))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        dataT1ce, dataFl, dataT1, dataT2 = self.datas[index]
        img0 = self.loader(dataT1ce)
        imgT1ce = img0[0].copy().astype(np.float32)

        img1 = self.loader(dataFl)
        imgFl = img1[0].copy().astype(np.float32)

        img2 = self.loader(dataT1)
        imgT1 = img2[0].copy().astype(np.float32)

        img3 = self.loader(dataT1)
        imgT2 = img3[0].copy().astype(np.float32)

        # 将data和datab1000进行通道合并,最多传3D传不了4D
        imgT = np.concatenate((imgT1, imgT1ce, imgFl, imgT2), 2)

        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        return imgT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


class MyDatasetFYY_NCCT(Dataset):
    '''Dataset类是读入数据集数据并且对读入的数据进行索引'''

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        super(MyDatasetFYY_NCCT, self).__init__()  # 对继承自父类的属性进行初始化
        dataList, labelList, addData1, addData2 = commonFunc.get_SpliteList(path, '_orgImg.nii', '_label.nii')
        datas = []
        for (i, j) in zip(dataList, labelList):
            datas.append((i, j))  # (图片信息，label)
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        '''用于按照索引读取每个元素的具体内容'''
        data, label = self.datas[index]
        img0 = self.loader(data)
        imgT = img0[0].copy().astype(np.float32)

        imgTag = self.loader(label)
        imgTagT = imgTag[0].copy().astype(np.float32)

        # imgTagT = np.multiply(imgTagT, imgT > 0)
        if self.transform is not None:
            imgT = self.transform(imgT)  # 数据标签转换为Tensor
        if self.target_transform is not None:
            imgTagT = self.target_transform(imgTagT)  # 数据标签转换为Tensor
        return imgT, imgTagT

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.datas)


# 测试主函数
if __name__ == "__main__":
    samplePath = r'E:\data\brainLesion\MR_3D\test\\'
    train_data = MyDatasetFYY_DWI(path=samplePath, transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader):
        x = data
        y = i
        x1 = torch.unsqueeze(x[0][:, 0:192, :, :], dim=1)
        x2 = torch.unsqueeze(x[0][:, 192:192 * 2, :, :], dim=1)
        x3 = torch.unsqueeze(x[0][:, 192 * 2:192 * 3, :, :], dim=1)
        img = np.concatenate((x1, x2, x3), 1)
        accP = IOU.IOU(x1, x2, '>', 0.5)
        z = 0
