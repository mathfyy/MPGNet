# import sys
import os
from SpecificApplication.brainLesion import trainBrainTumorMain_paper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    # brainTumor
    samplePath = r'/data1/Data/GBM/BrainTumor-pipeline/img/'
    savePath = r"/data1/Data/GBM/BrainTumor-pipeline/MPGNet/"
    trainBrainTumorMain_paper.trainBrainTumor(samplePath, savePath)

    print("over")

