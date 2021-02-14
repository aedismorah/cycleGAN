from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton, QComboBox

class ImageLabel(QLabel):
    def __init__(self, text, Type):
        super().__init__()

        border = 'dashed' if Type == 'input' else 'solid'
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n ' + text + ' \n\n')
        self.setStyleSheet('QLabel{border: 4px ' + border + ' #aaa}')
    
    def setPixmap(self, image):
        super().setPixmap(image)

def MainLayout(inputImg, outputImg, convertImg, modeChanged, deviceChanged):
    imageContainer = QWidget()
    imageContainer.setLayout(QHBoxLayout())

    imageContainer.layout().addWidget(inputImg)

    imageContainer.layout().addWidget(outputImg)

    actionContainer = QWidget()
    actionContainer.setLayout(QHBoxLayout())
    
    convertion = QPushButton('Convert', clicked = lambda: convertImg())
    
    device = QComboBox()
    device.addItems(['cpu', 'gpu'])
    device.currentTextChanged.connect(deviceChanged)

    mode = QComboBox()
    mode.addItems(['apple to orange', 'orange to apple', 'blonde to brunette', 'brunette to blonde', 'white to black', 'black to white'])
    mode.currentTextChanged.connect(modeChanged)

    actionContainer.layout().addWidget(convertion)
    actionContainer.layout().addWidget(device)
    actionContainer.layout().addWidget(mode)

    return imageContainer, actionContainer