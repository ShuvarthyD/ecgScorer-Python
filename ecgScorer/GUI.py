import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog,
    QVBoxLayout, QFrame, QLabel, QSpinBox, QPushButton, QLineEdit,
    QTableView, QDialog)
from PyQt6.QtCore import Qt, QAbstractTableModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
import pandas as pd
import ECGScorer
import os

class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Main
        self.setWindowTitle("ECGScorer")
        self.resize(850, 400)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Sections
        self.status_section = QLabel("<b><span style='font-size:12pt;'>Welcome To ecgScorer,<br> You can Load your Data</span></b>")
        self.status_section.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_section.setContentsMargins(10, 0, 10, 0)
        self.status_section.setStyleSheet("border: 1px solid #444;")
        self.dat_section = self.build_section("DATA", "#CCDEE6", input_type = "hz", buttons = ["Load Data", "Plot ECG"])
        self.qual_section = self.build_section("ECG_Quality", "#E6E6E6", buttons = ["ecgScore", "View Score"])
        self.exp_section = self.build_section("EXPORT", "#CCDEE6", buttons = ["export csv", "export xls"])
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        left_layout.addWidget(self.status_section, stretch = 2)
        left_layout.addWidget(self.dat_section, stretch = 3)
        left_layout.addWidget(self.qual_section, stretch = 2)
        left_layout.addWidget(self.exp_section, stretch = 2)
        main_layout.addLayout(left_layout, stretch = 2)
        
        self.plt_section = self.build_section("PLOT", "#CCDEE6", plot = True, status_lbl = True, slider = True)
        main_layout.addWidget(self.plt_section, stretch=4)
        
        # Buttons
        self.load_data.clicked.connect(self.load_file)
        self.plot_ecg.clicked.connect(self.update_plot)
        self.ecgscore.clicked.connect(self.ecg_score)
        self.view_score.clicked.connect(self.view_ecg_score)
        self.export_csv.clicked.connect(lambda: self.exp_report("csv"))
        self.export_xls.clicked.connect(lambda: self.exp_report("xls"))
        
    
    def build_section(self, title: str, color: str, plot: bool = False,
                      status_lbl: bool = False, buttons: list = [], slider: bool = False, input_type: str = "") -> QFrame:
        frame = QFrame()
        frame.setObjectName(title)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        frame.setStyleSheet(f"""
            QFrame#{title} {{
                background-color: {color};
                border: 1px solid #444;
            }}
        """)

        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title_label)

        if status_lbl:
            self.status_label = QLabel("<b><span style='font-size:20pt;'>__ecgScorer__</span></b>")
            self.status_label.setFixedHeight(50)
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status_label.setStyleSheet("""
                QLabel {
                    border: 1px solid #444;
                    background-color: #FFFFFF;
                }
            """)
            layout.addWidget(self.status_label)
        
        if input_type:
            input_section = QHBoxLayout()
            if input_type == "hz":
                input_lbl = QLabel("Sampling Frequency (Hz): ")
                self.input_hz =  QLineEdit()
                self.input_hz.setText("250")
                input_section.addWidget(input_lbl)
                input_section.addWidget(self.input_hz)
            layout.addLayout(input_section)
        
        if buttons:
            button_section = QHBoxLayout()
            for name in buttons:
                btn_name = name.lower().replace(" ", "_")
                button = QPushButton(name)
                button.setFixedWidth(100)
                button_section.addWidget(button)
                setattr(self, btn_name, button)
            layout.addLayout(button_section)
        
        if slider:
            signal_row = QHBoxLayout()

            signal_label = QLabel("Current Signal:")
            self.signal_spinner = QSpinBox()
            self.signal_spinner.setValue(0)
            self.signal_spinner.setFixedWidth(70) 
            self.signal_spinner.setMinimum(0)

            signal_row.addWidget(signal_label)
            signal_row.addWidget(self.signal_spinner)
            signal_row.addStretch()

            layout.addLayout(signal_row)

        if plot:
            self.figure, self.ax = plt.subplots()
            self.figure.subplots_adjust(left=0.2, bottom=0.2)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.ax.set_xlabel("Samples")
            self.ax.set_ylabel("Amplitude")
            self.ax.grid(True)
            layout.addWidget(self.canvas, stretch=10)
            
        return frame
    
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TXT file", "", "Text Files (*.txt *.csv *.xls *.xlsx)")
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".txt", ".csv"]:
                data = np.loadtxt(file_path, delimiter=",")
            elif ext == [".xls", ".xlsx"]:
                data = pd.read_excel(file_path).to_numpy()
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self.array_data = data
            self.signal_spinner.setMaximum(self.array_data.shape[1] - 1)
            self.status_section.setText("<b><span style='font-size:12pt;'>Data loaded Successfully, <br> Specify the Sampling Frequency</span></b>")
            self.status_label.setText("<b><span style='font-size:20pt;'>DATA LOADED SUCCESSFULLY</span></b>")

        except Exception:
            self.array_data = None
            self.status_section.setText("Incorrect File Format")
    
    def update_plot(self):
        if not hasattr(self, "array_data") or self.array_data is None:
            self.status_section.setText("No data to plot")
            return

        self.ax.clear()
        col_idx = self.signal_spinner.value() 
        self.ax.plot(self.array_data[:, col_idx])
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.canvas.draw()
    
    def ecg_score(self):
        if not hasattr(self, "array_data") or self.array_data is None:
            self.status_section.setText("No data loaded")
            return
        
        hz_str = self.input_hz.text()

        if not hz_str:
            self.status_section.setText("Input value is empty")
            return

        try:
            hz = int(hz_str) 
        except ValueError:
            self.status_section.setText("Input value must be a integer")
            return

        label, comment, signalNum, aScore, iScore = ECGScorer.scorer12(self.array_data,hz)
        df = pd.DataFrame({
            'signalNum': signalNum,
            'class': label,
            'comment': comment,
            'aScore': aScore,
            'iScore': iScore
        })

        self.score_report = df
        self.status_section.setText("<b><span style='font-size:12pt;'>******************* <br> Calcuation Completed <br> You can Export the Result,</span></b>")
        self.status_label.setText("<b><span style='font-size:20pt;'>CALCUATION COMPLETED</span></b>")
            
    def view_ecg_score(self):
        if not hasattr(self, "score_report") or self.score_report is None:
            self.status_section.setText("No result data to show")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Score Report")

        layout = QVBoxLayout()
        table_view = QTableView()
        model = PandasModel(self.score_report)
        table_view.setModel(model)
        table_view.resizeColumnsToContents()
        layout.addWidget(table_view)
        dialog.setLayout(layout)
        dialog.resize(600, 400)
        dialog.exec()
    
    def exp_report(self, type):
        if not hasattr(self, "score_report") or self.score_report is None:
            self.status_section.setText("No result data to export")
            return

        if type == "csv":
            filter_str = "CSV Files (*.csv)"
        elif type == "xls":
            filter_str = "Excel Files (*.xlsx)"

        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {type.upper()}", "", filter_str)
        if not file_path:
            return
        
        try:
            if type == "csv":
                self.score_report.to_csv(file_path, index=False)
            elif type == "xls":
                self.score_report.to_excel(file_path, index=False)
            self.status_section.setText("<b><span style='font-size:12pt;'>******************* <br> Result Exported Successfully <br> *******************</span></b>")
        except Exception:
            self.status_section.setText("Failed to export")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())