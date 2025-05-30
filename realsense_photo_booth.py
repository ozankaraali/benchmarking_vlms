import sys
import os
import time
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QCheckBox, QGroupBox, QFormLayout
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

import pyrealsense2 as rs
import numpy as np

class RealSenseCapture:
    def __init__(self, width=640, height=480, fps=30, enable_depth=False):
        self.pipeline = rs.pipeline()
        self.enable_depth = enable_depth
        config = rs.config()

        try:
            # List available devices
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found!")

            device = devices[0]
            print(f"\nFound RealSense device: {device.get_info(rs.camera_info.name)}")

            # Query device sensors
            sensors = device.query_sensors()
            print("\nAvailable sensors:")
            for sensor in sensors:
                print(f"  {sensor.get_info(rs.camera_info.name)}")
                print("  Supported formats:")
                for profile in sensor.get_stream_profiles():
                    if profile.is_video_stream_profile():
                        video_profile = profile.as_video_stream_profile()
                        fmt = profile.format()
                        print(
                            f"    {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps ({fmt})")

            # Configure streams with passed parameters
            print(f"\nConfiguring camera for {width}x{height} @ {fps}fps...")
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if enable_depth:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

            # Start pipeline
            pipeline_profile = self.pipeline.start(config)

            # Get device sensors after pipeline start
            device = pipeline_profile.get_device()
            sensors = device.query_sensors()

            # Find and configure the RGB sensor
            rgb_sensor = None
            for sensor in sensors:
                if 'RGB' in sensor.get_info(rs.camera_info.name):
                    rgb_sensor = sensor
                    break
                elif 'Stereo' in sensor.get_info(rs.camera_info.name):
                    rgb_sensor = sensor
                    break

            if rgb_sensor:
                try:
                    # Enable auto exposure and auto white balance
                    # rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    # rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                    print("\nCamera settings:")
                    # print(f"Auto exposure enabled: {rgb_sensor.get_option(rs.option.enable_auto_exposure)}")
                    # print(f"Auto white balance enabled: {rgb_sensor.get_option(rs.option.enable_auto_white_balance)}")
                except Exception as e:
                    print(f"Warning: Could not configure camera settings: {e}")

            print("\nRealSense camera initialized successfully")

            # Warm up the camera
            for _ in range(5):
                self.pipeline.wait_for_frames(1000)

        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            raise

    def get_frames(self, timeout_ms=1000):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                return None, None

            color_frame = frames.get_color_frame()

            if not color_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())

            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    return None, None
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                depth_image = None

            return color_image, depth_image

        except Exception as e:
            print(f"Error getting frames from camera: {e}")
            return None, None

    def release(self):
        try:
            self.pipeline.stop()
            print("RealSense camera released")
        except Exception as e:
            print(f"Error releasing camera: {e}")

class RealSensePhotoBooth(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense Photo Booth")
        self.setGeometry(100, 100, 1000, 700)

        # Create output directory
        self.output_dir = "photo_booth_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create "new" directory in output directory to save RGB's.
        self.rgb_output_dir = os.path.join(self.output_dir, "new")
        os.makedirs(self.rgb_output_dir, exist_ok=True)

        # Initialize RealSense camera with depth enabled
        self.camera = RealSenseCapture(width=1280, height=720, fps=30, enable_depth=True)

        # Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Preview area
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.preview_label)

        # Depth preview (optional)
        self.depth_preview_label = QLabel()
        self.depth_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.depth_preview_label)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()

        # Resolution settings
        self.resolution_label = QLabel("1280x720")
        settings_layout.addRow("Resolution:", self.resolution_label)

        # Show depth checkbox
        self.show_depth_checkbox = QCheckBox("Show Depth Preview")
        self.show_depth_checkbox.setChecked(True)
        self.show_depth_checkbox.stateChanged.connect(self.toggle_depth_preview)
        settings_layout.addRow(self.show_depth_checkbox)

        settings_group.setLayout(settings_layout)
        controls_layout.addWidget(settings_group)

        # Button layout
        button_layout = QVBoxLayout()

        # Capture button
        self.capture_button = QPushButton("Capture Photo (PNG + Depth)")
        self.capture_button.clicked.connect(self.capture_photo)
        self.capture_button.setMinimumHeight(50)
        button_layout.addWidget(self.capture_button)

        controls_layout.addLayout(button_layout)
        self.main_layout.addLayout(controls_layout)

        # Timer for updating preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)  # Update every 30ms

        # Initialize depth preview visibility
        self.toggle_depth_preview()

    def toggle_depth_preview(self, state=None):
        """Toggle the visibility of the depth preview"""
        show_depth = self.show_depth_checkbox.isChecked() if state is None else state
        self.depth_preview_label.setVisible(show_depth)

    def update_preview(self):
        """Update the color and depth preview images"""
        # Get frames from RealSense camera
        color_image, depth_image = self.camera.get_frames()

        if color_image is None:
            return

        # Convert color image to QImage and display
        h, w, ch = color_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(color_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # If depth is enabled and available, display it
        if depth_image is not None and self.show_depth_checkbox.isChecked():
            # Normalize depth for display (convert to 8-bit grayscale)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            h, w, ch = depth_colormap.shape
            bytes_per_line = ch * w
            depth_qt_image = QImage(depth_colormap.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.depth_preview_label.setPixmap(QPixmap.fromImage(depth_qt_image).scaled(
                self.depth_preview_label.width(), self.depth_preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def capture_photo(self):
        """Capture and save color image as PNG and depth image as grayscale PNG"""
        # Get frames from RealSense camera
        color_image, depth_image = self.camera.get_frames()

        if color_image is None:
            return

        # Create timestamp for filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save color image as lossless PNG
        color_filename = f"{self.rgb_output_dir}/photo_{timestamp}.png"
        cv2.imwrite(color_filename, color_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = lossless

        # If depth is available, save it as a separate grayscale PNG
        if depth_image is not None:
            # Normalize depth for better visualization (16-bit to 8-bit)
            # Scale depth to use full 16-bit range for better detail
            depth_normalized = cv2.normalize(depth_image, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

            # Save as 16-bit PNG
            depth_filename = f"{self.output_dir}/photo_{timestamp}_depth.png"
            cv2.imwrite(depth_filename, depth_normalized, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Also save a visualization that's easier for vision models to understand
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            depth_vis_filename = f"{self.output_dir}/photo_{timestamp}_depth_vis.png"
            cv2.imwrite(depth_vis_filename, depth_vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f"Photos saved: \n- Color: {color_filename}\n- Depth: {depth_filename}\n- Depth Visualization: {depth_vis_filename}")
        else:
            print(f"Photo saved: {color_filename}")

    def closeEvent(self, event):
        """Handle window close event"""
        # Release the camera resources
        self.camera.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealSensePhotoBooth()
    window.show()
    sys.exit(app.exec())