import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout

from cycleGAN.generator import Generator

import torch
from torchvision import transforms
import numpy as np

from cycleGAN.generator import Generator

from interface.widget import ImageLabel, MainLayout
from interface.transforms import QPixmapToNumpy, NumpyToQPixmap

class ImgConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(640, 380)
        self.setAcceptDrops(True)

        self.device = 'cpu'
        self.transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(256)])
        self.imgArray = -1

        self.generator = Generator()
        self.loadGenWeights('apple to orange')

        mainLayout = QGridLayout()

        self.inputImg = ImageLabel('Drop Image Here', 'input')
        self.outputImg = ImageLabel('Output', 'output')

        imageContainer, actionContainer = MainLayout(self.inputImg, self.outputImg, self.convertImg, self.modeChanged, self.deviceChanged)

        mainLayout.addWidget(imageContainer, 1, 1, 8, 1)
        mainLayout.addWidget(actionContainer, 9, 1, 1, 1)

        self.setLayout(mainLayout)
    
    def deviceChanged(self, value):
        self.device = torch.device('cuda:0') if value == 'gpu' else torch.device('cpu')
        self.generator.to(self.device)
        print('using ' + value)

    def modeChanged(self, value):
        self.loadGenWeights(value)
        print('loaded ' + value)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            filepath = event.mimeData().urls()[0].toLocalFile()
            self.setImage(filepath)

            event.accept()
        else:
            event.ignore()

    def setImage(self, filepath):
        if filepath.split('.')[-1] == 'jpg' or filepath.split('.')[-1] == 'png' or filepath.split('.')[-1] == 'jpeg':
            self.imgArray = QPixmapToNumpy(QPixmap(filepath))[:, :, :3]
            npImg = np.array(torch.abs(self.transform(self.imgArray.copy())).permute(1, 2, 0).numpy() * 255, dtype=np.uint8).copy()
            self.inputImg.setPixmap(QPixmap.fromImage(NumpyToQPixmap(npImg).copy()))
            print('Image acquired')
        else:
            print('A file is not an image')

    def loadGenWeights(self, mode):
        filename = mode.replace(' ', '_')
        self.generator.load_state_dict(torch.load('weights/' + filename))

    def convertImg(self):
        if type(self.imgArray) is np.ndarray:
            npImg = np.array(self.imgArray[:, :, :3].copy(), dtype=np.uint8)
            npImg[:, :, 0], npImg[:, :, 2] = self.imgArray[:, :, 2], self.imgArray[:, :, 0]

            tensorImg = self.transform(npImg.copy()).unsqueeze(0).to(self.device)
            tensorImg = self.generator(tensorImg).squeeze(0).detach().to('cpu')
            output = np.array(torch.abs(tensorImg.permute(1, 2, 0) * 255), dtype=np.uint8).copy()

            self.outputImg.setPixmap(QPixmap.fromImage(NumpyToQPixmap(output, False).copy()))

app = QApplication(sys.argv)
demo = ImgConverter()
demo.show()
sys.exit(app.exec_())