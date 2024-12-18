import scipy.io as scio


def MATRead(dir):
    data = scio.loadmat(dir)
    return data


def MATWrite(dir, data, sTag):
    scio.savemat(dir, {sTag: data[sTag]})
    return


if __name__ == "__main__":
    data = MATRead(r'E:\data\HT\data\netData\sample\0000048692_vecorPre.mat')
    MATWrite(r'E:\data\HT\data\netData\sample\0000048692_vecorPre11.mat', data, 'vector')
    x = 1
