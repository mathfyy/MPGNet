import os
import SimpleITK as sitk
import torch


def writeArrayToNii(img, savePath, name):
    img = img.to("cpu")
    img = img.detach().numpy()
    result = sitk.GetImageFromArray(img)
    sitk.WriteImage(result, os.path.join(savePath, name))
    return result
