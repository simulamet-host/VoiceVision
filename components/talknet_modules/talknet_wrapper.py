import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
import logging

from components.loss_functions import lossAV, lossA, lossV
from components.talknet_modules.talknet_model import talkNetModel
from utils import device

log = logging.getLogger('aiproducer')

class talkNet(nn.Module):
    def __init__(self):
        super(talkNet, self).__init__()
        self.model = talkNetModel().to(device)
        self.lossAV = lossAV().to(device)
        self.lossA = lossA().to(device)
        self.lossV = lossV().to(device)
        log.info("%s Model parameter count = %.2f MB",
                        time.strftime("%m-%d %H:%M:%S"),
                        sum(param.numel() for param in self.model.parameters()) / 1024 / 1024)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=device)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    log.warning("%s is not in the model.", origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                log.error("Wrong parameter length: %s, model: %s, loaded: %s",
                          origName, selfState[name].size(), loadedState[origName].size())
                continue
            selfState[name].copy_(param)
