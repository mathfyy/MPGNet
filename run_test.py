import os
import time
import torch

from SpecificApplication.brainLesion import testBrainTumor_paper

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':

    inputSize = [24, 256, 256]
    sizeG = [6, 8, 8]
    sim_thr = 0.75

    samplePath = r'/data1/Data/GBM/BrainTumor-pipeline/img/'
    # savePath = r"/data1/Data/GBM/BrainTumor-pipeline/MPGNet_loss_all/MPGNet_201/"
    # savePath = r"/data1/Data/GBM/BrainTumor-pipeline/cross_ret/kMaX-DeepLab_GCN_new-20-1.5/"
    savePath = r"/data1/Data/GBM/BrainTumor-pipeline/cross_ret_use_removeskull/MPGNet/"

    model_index = 200

    since = time.time()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = torch.load(savePath + '1_net_val.pkl')
    # net = net.to(device)
    # net.eval()
    # from thop import profile
    # data = torch.randn([4, 4, 24, 256, 256]).to(device)
    # flops, params = profile(net, inputs=(data,))
    # print(flops / 1e9, params / 1e6)

    for i in range(5):
        testBrainTumor_paper.testBrain(samplePath, savePath, inputSize, model_index, sizeG, sim_thr, i + 1)

    # testBrainTumor_paper.testBrain(samplePath, savePath, inputSize, model_index, sizeG, sim_thr, 1)
    # testBrainTumor_paper.savePreResult(samplePath, savePath, inputSize, model_index, sizeG, sim_thr)

    time_elapsed = time.time() - since
    print("Time in %f" % (time_elapsed))

    print("over")