import os


def get_Filelist(folder_path, str):
    # 获取文件夹下指定后缀的所有文件的文件名列表
    # folder_path:文件夹路径
    # str:要遍历的数据格式，例如：'.mat'
    filename_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.find(str) >= 0:
                filename_list.append(os.path.join(root, file))
    return filename_list


def get_SpliteList(folder_path, strData, strLabel, strAdd1='', strAdd2=''):
    data = []
    label = []
    addData1 = []
    addData2 = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.find(strData) >= 0:
                name = file[0:file.find(strData)]
                data.append(os.path.join(root, name + strData))
                label.append(os.path.join(root, name + strLabel))
                addData1.append(os.path.join(root, name + strAdd1))
                addData2.append(os.path.join(root, name + strAdd2))
            # if file.find(strData) >= 0:
            #     data.append(os.path.join(root, file))
            # if file.find(strLabel) >= 0:
            #     label.append(os.path.join(root, file))
            # if file.find(strAdd1) >= 0:
            #     addData1.append(os.path.join(root, file))
            # if file.find(strAdd2) >= 0:
            #     addData2.append(os.path.join(root, file))
    return data, label, addData1, addData2


def get_SpliteList3(folder_path, strData, strLabel, strAdd1='', strAdd2='', strAdd3='', strAdd4=''):
    data = []
    label = []
    addData1 = []
    addData2 = []
    addData3 = []
    addData4 = []

    log = open(folder_path + "_name.txt", "w")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.find(strData) >= 0:
                name = file[0:file.find(strData)]
                log.write("%s\n" % (name))
                data.append(os.path.join(root, name + strData))
                label.append(os.path.join(root, name + strLabel))
                addData1.append(os.path.join(root, name + strAdd1))
                addData2.append(os.path.join(root, name + strAdd2))
                addData3.append(os.path.join(root, name + strAdd3))
                addData4.append(os.path.join(root, name + strAdd4))
            # if file.find(strData) >= 0:
            #     data.append(os.path.join(root, file))
            # if file.find(strLabel) >= 0:
            #     label.append(os.path.join(root, file))
            # if file.find(strAdd1) >= 0:
            #     addData1.append(os.path.join(root, file))
            # if file.find(strAdd2) >= 0:
            #     addData2.append(os.path.join(root, file))
            # if file.find(strAdd3) >= 0:
            #     addData3.append(os.path.join(root, file))
            # if file.find(strAdd3) >= 0:
            #     addData4.append(os.path.join(root, file))
    return data, label, addData1, addData2, addData3, addData4


def get_SpliteList_txt(folder_path, txt_path, strData, strLabel, strAdd1='', strAdd2='', strAdd3='', strAdd4=''):
    data = []
    label = []
    addData1 = []
    addData2 = []
    addData3 = []
    addData4 = []

    with open(folder_path + txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.__sizeof__() > 0:
                name = line.strip()
                data.append(os.path.join(folder_path, name + strData))
                label.append(os.path.join(folder_path, name + strLabel))
                addData1.append(os.path.join(folder_path, name + strAdd1))
                addData2.append(os.path.join(folder_path, name + strAdd2))
                addData3.append(os.path.join(folder_path, name + strAdd3))
                addData4.append(os.path.join(folder_path, name + strAdd4))
    return data, label, addData1, addData2, addData3, addData4
