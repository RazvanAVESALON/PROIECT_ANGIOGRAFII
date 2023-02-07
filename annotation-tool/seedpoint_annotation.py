import os
import sys

import json

import numpy as np

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


def apply_transfer_function(frames, w, c):
    y_min = 0.0
    y_max = 255.0

    below = (frames <= c - 0.5 - (w-1) / 2.0)
    above = (frames > c - 0.5 + (w-1) / 2.0)
    between = (~below) & (~above)

    result = frames.copy()

    result[below] = y_min
    result[above] = y_max
    result[between] = ((frames[between] - (c - 0.5)) /
                       (w-1) + 0.5) * (y_max - y_min) + y_min

    return result


class SeedpointAnnotationWidow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi("ui_files/seedpoint_annotation.ui", self)
        self.show()

        self.angio_view.mousePressEvent = self.on_image_view_mousePressEvent
        self.angio_view.wheelEvent = self.on_image_view_wheelEvent

        self.angio = None
        self.clipping_points = {}
        self.current_frame = 0
        self.last_opened_path = r""

    @pyqtSlot()
    def on_open_btn_clicked(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open study", self.last_opened_path)
        if path:
            self.last_opened_path = path
            self.cases = []
            for dir in os.listdir(path):
                dirpath = os.path.join(path, dir)
                if os.path.isdir(dirpath):
                    self.cases.append(dirpath)

            if len(self.cases) > 0:
                self.current_case_idx = 0
                self.load_case(self.cases[self.current_case_idx])
            else:
                print("No cases found. Make sure you select the study directory.")

    @pyqtSlot()
    def on_clear_frame_btn_clicked(self):
        if str(self.current_frame) in self.clipping_points:
            del self.clipping_points[str(self.current_frame)]
            self.update_image_view()

    @pyqtSlot()
    def on_previous_frame_btn_clicked(self):
        if self.angio is None:
            return

        self.current_frame -= 1
        self.current_frame %= self.angio.shape[0]
        self.update_image_view()

    @pyqtSlot()
    def on_next_frame_btn_clicked(self):
        if self.angio is None:
            return

        self.current_frame += 1
        self.current_frame %= self.angio.shape[0]
        self.update_image_view()

    @pyqtSlot()
    def on_previous_case_btn_clicked(self):
        with open(self.annotation_path, "w") as f:
            json.dump(self.clipping_points, f, indent=2, sort_keys=True)

        self.current_case_idx -= 1
        self.current_case_idx %= len(self.cases)
        self.load_case(self.cases[self.current_case_idx])

    @pyqtSlot()
    def on_next_case_btn_clicked(self):
        with open(self.annotation_path, "w") as f:
            json.dump(self.clipping_points, f, indent=2, sort_keys=True)

        self.current_case_idx += 1
        self.current_case_idx %= len(self.cases)
        self.load_case(self.cases[self.current_case_idx])

    def on_image_view_mousePressEvent(self, event):
        if self.distal_pts_btn.isChecked():
            pos = (event.pos().y(), event.pos().x())
            self.clipping_points[str(self.current_frame)] = pos
            self.update_image_view()

    def on_image_view_wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.on_previous_frame_btn_clicked()
        else:
            self.on_next_frame_btn_clicked()

    def on_image_view_mouseMoveEvent(self, event):
        pass

    def on_image_view_mouseReleaseEvent(self, event):
        pass

    def load_case(self, case_path):
        with open(os.path.join(case_path, "angio_loader_header.json"), "r") as f:
            self.metadata = json.load(f)

        self.case_name_edit.setText(case_path)

        self.annotation_path = os.path.join(case_path, "clipping_points.json")

        with open(self.annotation_path, "r") as f:
            self.clipping_points = json.load(f)

        self.angio = np.load(os.path.join(
            case_path, "frame_extractor_frames.npz"))["arr_0"]
        self.angio = apply_transfer_function(
            self.angio, self.metadata['WindowWidth'], self.metadata['WindowCenter'])
        self.angio = self.angio.astype(np.uint8)

        self.current_frame = 0

        self.update_image_view()

    def update_image_view(self):
        if self.angio is None:
            return

        self.slider.setMaximum(self.angio.shape[0] - 1)
        self.slider.setValue(self.current_frame)

        qimage = QImage(self.angio[self.current_frame].data, self.metadata["ImageSize"][0],
                        self.metadata["ImageSize"][1], self.metadata["ImageSize"][0], QImage.Format_Grayscale8)
        self.angio_view.setFixedWidth(self.metadata["ImageSize"][1])
        self.angio_view.setFixedHeight(self.metadata["ImageSize"][0])
        self.angio_view.setPixmap(QPixmap(qimage))

        if str(self.current_frame) in self.clipping_points:
            painter = QPainter(self.angio_view.pixmap())
            painter.setPen(QColor("yellow"))
            painter.setBrush(QColor("yellow"))

            painter.drawEllipse(QPoint(self.clipping_points[str(
                self.current_frame)][1], self.clipping_points[str(self.current_frame)][0]), 5, 5)

            painter.end()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeedpointAnnotationWidow()
    sys.exit(app.exec_())
