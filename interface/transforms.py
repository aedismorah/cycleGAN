from PyQt5.QtGui import QImage
import numpy as np

def QPixmapToNumpy(QPixmapObject):
    Qimg = QPixmapObject.toImage()
    height, width = Qimg.size().height(), Qimg.size().width()
    stringImg = Qimg.bits().asstring(height * width * 4)
    return np.frombuffer(stringImg, dtype=np.uint8).reshape((height, width, 4))

def NumpyToQPixmap(npImg, permuteChannels=True):
        imgCopy = np.copy(npImg)
        if permuteChannels:
            imgCopy[:, :, 0], imgCopy[:, :, 2] = npImg[:, :, 2], npImg[:, :, 0]
        return QImage(imgCopy.data, imgCopy.shape[1], imgCopy.shape[0], QImage.Format_RGB888)