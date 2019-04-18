import sys
from PyQt5.QtWidgets import QApplication
from mainwindow import MyMnistWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()
