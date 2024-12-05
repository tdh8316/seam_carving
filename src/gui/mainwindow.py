import sys

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QFileDialog,
    QLineEdit,
    QApplication,
    QGroupBox,
    QComboBox,
    QProgressBar,
    QMessageBox,
)

from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder
from src.lib import Image, CarvableImage


class AppStyles:
    """
    Centralized styles for the application.
    """

    WINDOW_STYLE = """
        QMainWindow {
            background-color: #1E1E2F; /* Darker, modern background */
            color: #FFFFFF; /* Default text color */
        }
    """

    BUTTON_STYLE = """
        QPushButton {
            background-color: #0066CC; /* Modern vibrant blue */
            color: #FFFFFF; /* White text for contrast */
            border: 2px solid #004C99; /* Stronger border for definition */
            border-radius: 10px; /* Rounded edges for a sleek look */
            padding: 10px 20px; /* Reduced padding for a smaller button */
            font-size: 14px; /* Slightly smaller font size */
            font-weight: 500; /* Balanced text weight */
            font-family: 'Segoe UI', Arial, sans-serif; /* Clean and modern font */
        }
        QPushButton:hover {
            background-color: #004C99; /* Darker blue on hover */
        }
        QPushButton:pressed {
            background-color: #003366; /* Deep blue for pressed state */
        }
        QPushButton:disabled {
            background-color: #A6A6A6; /* Gray background for disabled button */
            color: #E0E0E0; /* Light text for contrast */
            border: 1px solid #7A7A7A; /* Subtle border for disabled state */
        }
    """

    DROP_DOWN_STYLE = """
        QComboBox {
            background-color: #2D2D3C; /* Subtle dark gray */
            color: #FFFFFF;
            border: 1px solid #444455; /* Soft border for contrast */
            border-radius: 8px;
            padding: 11px;
            font-size: 14px;
            font-family: Arial, sans-serif;
            min-width: 150px;
        }
        
        QComboBox::drop-down {
            border: 0;
            background-color: #2D2D3C;
            width: 30px;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        
        QComboBox::down-arrow {
            image: url("./src/gui/assets/down_arrow.png");
            width: 16px;
            height: 16px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #3C3C4F; /* Dropdown background */
            color: #FFFFFF;
            border: 1px solid #555566;
            selection-background-color: #444466; /* Selection color */
            font-size: 14px;
        }
    """

    LINE_EDIT_STYLE = """
        QLineEdit {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #CCCCCC; /* Subtle light gray border */
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        QLineEdit:focus {
            border-color: #0078D4; /* Blue border on focus */
            outline: none;
        }
    """

    IMAGE_LABEL_STYLE = """
        QLabel {
            border: 2px solid #444455;
            background-color: #2D2D3C;
            color: #FFFFFF;
            font-size: 16px;
            font-family: Arial, sans-serif;
            border-radius: 8px;
            padding: 8px;
        }
    """

    GROUP_BOX_STYLE = """
        QGroupBox {
            background-color: #2D2D3C; /* Dark gray background */
            color: #FFFFFF;
            border: 1px solid #444455; /* Subtle border */
            border-radius: 8px;
            padding: 15px;
            font-weight: bold;
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px;
            background-color: transparent; /* Match group box background */
            color: #FFFFFF;
        }
    """


class MainWindow(QMainWindow):
    """
    Main window for the application. Contains the main UI elements and logic.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seam Carving")
        self.setGeometry(200, 200, 1000, 800)
        self.setStyleSheet(AppStyles.WINDOW_STYLE)

        self.original_image = None
        self.carved_image = None
        self.export_button = None

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add other UI components
        panel_hbox = QHBoxLayout()

        controls_group = self._create_controls_group()
        panel_hbox.addWidget(controls_group)

        enlarge_controls_group = self._create_enlarge_controls_group()
        panel_hbox.addWidget(enlarge_controls_group)

        main_layout.addLayout(panel_hbox)

        image_display_group = self._create_image_display_group()
        main_layout.addWidget(image_display_group)

        # Add Export Button
        export_layout = QHBoxLayout()
        export_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.export_button = QPushButton("Export Image")
        self.export_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.export_button.clicked.connect(self.export_image)

        export_layout.addWidget(self.export_button)

        main_layout.addLayout(export_layout)

        # Add Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

    def _create_controls_group(self):
        controls_group = QGroupBox("Image Controls")
        controls_group.setStyleSheet(AppStyles.GROUP_BOX_STYLE)
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load Button
        self.load_button = QPushButton("Load Image")
        self.load_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.load_button.clicked.connect(self.load_image)

        # Aspect Ratio
        self.aspect_ratio_dropdown = QComboBox()
        self.aspect_ratio_dropdown.addItems(
            ["16:9", "4:5", "1:1", "3:4", "9:16", "Custom"]
        )
        self.aspect_ratio_dropdown.setStyleSheet(AppStyles.DROP_DOWN_STYLE)

        # Seam Input
        self.seams_input = QLineEdit()
        self.seams_input.setPlaceholderText("Enter number of seams to carve")
        self.seams_input.setStyleSheet(AppStyles.LINE_EDIT_STYLE)

        # Carve Button
        self.carve_button = QPushButton("Resize Image")
        self.carve_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.carve_button.clicked.connect(self.start_seam_carving)  # st_seam_carving
        ##############################

        # Add to layout
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.aspect_ratio_dropdown)
        # button_layout.addWidget(self.seams_input) # seams_input enter box
        button_layout.addWidget(self.carve_button)
        controls_layout.addLayout(button_layout)
        return controls_group

    def _create_enlarge_controls_group(self):
        controls_group = QGroupBox("Enlarge Controls")
        controls_group.setMaximumWidth(500)
        controls_group.setStyleSheet(AppStyles.GROUP_BOX_STYLE)
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Seam Input - Width
        self.seams_input_width = QLineEdit()
        self.seams_input_width.setMaximumWidth(100)
        self.seams_input_width.setPlaceholderText("Width")
        self.seams_input_width.setStyleSheet(AppStyles.LINE_EDIT_STYLE)

        # Seam Input - Height
        self.seams_input_height = QLineEdit()
        self.seams_input_height.setMaximumWidth(100)
        self.seams_input_height.setPlaceholderText("Height")
        self.seams_input_height.setStyleSheet(AppStyles.LINE_EDIT_STYLE)

        # Carve Button
        self.carve_button = QPushButton("Enlarge Image")
        self.carve_button.setMaximumWidth(150)
        self.carve_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.carve_button.clicked.connect(self.start_seam_enlarge)  # st_seam_carving
        ##############################

        # Add to layout
        button_layout.addWidget(self.seams_input_width)
        button_layout.addWidget(self.seams_input_height)
        button_layout.addWidget(self.carve_button)
        controls_layout.addLayout(button_layout)
        return controls_group

    def _create_image_display_group(self):
        image_display_group = QGroupBox("Image Comparison")
        image_display_group.setStyleSheet(AppStyles.GROUP_BOX_STYLE)
        image_layout = QHBoxLayout()
        image_display_group.setLayout(image_layout)

        self.original_image_label = self._create_image_label()
        self.carved_image_label = self._create_image_label()

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.carved_image_label)
        return image_display_group

    def _create_image_label(self):
        image_label = QLabel(self)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet(AppStyles.IMAGE_LABEL_STYLE)
        image_label.setText("No Image Loaded")  # Placeholder text
        image_label.setMinimumSize(700, 400)  # fixed image label size

        return image_label

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)"
        )
        if file_path:
            self.original_image = Image.from_path(file_path)
            self._display_image(self.original_image.mat, self.original_image_label)

    def export_image(self):
        if not hasattr(self, "final_image"):
            QMessageBox.warning(
                self,
                "Error",
                "No image to export. Please load an image and resize it first.",
            )
            return

        assert isinstance(
            self.final_image, Image
        ), "Final image must be an Image object."

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image as", "", "Images (*.png *.xpm *.jpg *.bmp)"
        )
        if file_path:
            print(file_path)
            self.final_image.save(file_path)

    def _add_progress_bar(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        return self.progress_bar

    @staticmethod
    def ratio_to_num_seams(original_width, original_height, aspect_ratio):
        """
        Convert an aspect ratio to the number of seams required to resize an image.

          Args:
            original_width (int): The width of the original image.
            original_height (int): The height of the original image.
            aspect_ratio (str): The desired aspect ratio in the format "width:height" (e.g., "16:9").

          Returns:
            tuple: A tuple containing the number of vertical seams and horizontal seams (vertical_seams, horizontal_seams).
        """

        try:
            width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
        except ValueError:
            raise ValueError(
                "Aspect ratio must be in the format 'width:height', e.g., '16:9'."
            )

        # Calculate the target dimensions
        target_width = int(original_height * (width_ratio / height_ratio))
        target_height = int(original_width * (height_ratio / width_ratio))

        # Adjust target dimensions
        target_width = min(target_width, original_width)
        target_height = min(target_height, original_height)

        print(f"Target dimensions: Width={target_width}, Height={target_height}")

        # Calculate the number of seams
        vertical_seams = max(0, original_width - target_width)
        horizontal_seams = max(0, original_height - target_height)

        print(
            f"Calculated seams: Vertical={vertical_seams}, Horizontal={horizontal_seams}"
        )

        return vertical_seams, horizontal_seams

    def start_seam_carving(self):
        if not self.original_image:
            print("No image loaded.")
            return
        self.carved_image_label.setText("Processing...")
        self.carved_image_label.repaint()
        QApplication.processEvents()

        try:
            aspect_ratio = self.aspect_ratio_dropdown.currentText()
            print(f"aspect_ratio: {aspect_ratio}")

            original_height, original_width = self.original_image.mat.shape[:2]
            print(f"width: {original_width} , height: {original_height}")

            # Convert aspect ratio to number of seams
            num_v_seams, num_h_seams = self.ratio_to_num_seams(
                original_width, original_height, aspect_ratio
            )
            print(num_v_seams, num_h_seams)

            self.progress_bar.setValue(0)

        except ValueError:
            print("Please enter a valid integer for seams.")
            return

        # Vertical seam carving (reduce width)
        try:
            carvable_image = CarvableImage(self.original_image)
            carvable_image.energy_function = EnergyCalculator.squared_diff
            carvable_image.seam_function = SeamFinder.find_seam

            for i in range(1, num_v_seams + 1):
                carvable_image.seam_carve_with_mask(1)  # Carve one seam at a time
                self.progress_bar.setValue(
                    int((i / num_v_seams) * 50)
                )  # Update progress bar for vertical seams

            carved_data = carvable_image.seam_carve_with_mask(num_v_seams).img.mat
            self.vertical_save = Image(carved_data)

        except Exception as e:
            print(f"Error in vertical seam carving: {e}")
            return

        # Horizontal seam carving (reduce height)
        try:
            carvable_image_hor = CarvableImage(self.vertical_save)
            carvable_image_hor.img.mat = cv2.rotate(
                carvable_image_hor.img.mat, cv2.ROTATE_90_CLOCKWISE
            )

            # Perform seam carving one seam at a time and update the progress bar
            for i in range(1, num_h_seams + 1):
                carvable_image_hor.seam_carve_with_mask(1)  # Carve one seam at a time
                self.progress_bar.setValue(
                    50 + int((i / num_h_seams) * 50)
                )  # Update progress bar for horizontal seams

            # Rotate the final result back to the original orientation
            carved_data_hor = cv2.rotate(
                carvable_image_hor.img.mat, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        except Exception as e:
            print(f"Error in horizontal seam carving: {e}")
            return

        try:
            self.final_image = Image(carved_data_hor)
            self._display_image(self.final_image.mat, self.carved_image_label)

            self.progress_bar.setValue(100)  # Set progress to 100% when complete
        except Exception as e:
            print(f"Error in displaying the carved image: {e}")
            return

    def start_seam_enlarge(self):
        if not self.original_image:
            print("No image loaded.")
            return

        try:
            width_pixel = int(self.seams_input_width.text())
            height_pixel = int(self.seams_input_height.text())
            print(f"width: {width_pixel} , height: {height_pixel}")

        except ValueError:
            print("Please enter a valid integer for seams.")
            return

        # Vertical seam carving (enlarge width)
        try:
            enlarge_image = CarvableImage(self.original_image)
            enlarge_image.energy_function = EnergyCalculator.squared_diff
            enlarge_image.seam_function = SeamFinder.find_seam
            enlarged_data = enlarge_image.seam_carve_enlarge(width_pixel).img.mat

        except Exception as e:
            print(f"Error in vertical seam carving: {e}")
            return

        enlarged_vertical_save = Image(enlarged_data)

        # Horizontal seam carving (enlarge height)
        try:
            enlarge_image_hor = CarvableImage(enlarged_vertical_save)
            enlarge_image_hor.img.mat = cv2.rotate(
                enlarge_image_hor.img.mat, cv2.ROTATE_90_CLOCKWISE
            )
            enlarge_image_hor.energy_function = EnergyCalculator.squared_diff
            enlarge_image_hor.seam_function = SeamFinder.find_seam
            enlarged_data_hor = enlarge_image_hor.seam_carve_enlarge(
                height_pixel
            ).img.mat
            enlarged_data_hor = cv2.rotate(
                enlarged_data_hor, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        except Exception as e:
            print(f"Error in horizontal seam carving: {e}")
            return

        try:
            self.final_image = Image(enlarged_data_hor)
            self._display_image(self.final_image.mat, self.carved_image_label)
        except Exception as e:
            print(f"Error in displaying the carved image: {e}")
            return

    def _display_image(self, image_data, label: QLabel):
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format.Format_BGR888
        )

        pixmap = QPixmap.fromImage(q_img)

        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            label_size.width(),
            label_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        label.setPixmap(scaled_pixmap)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
