from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec


class VOVisualizer:
    """Handles real-time visualization of VO pipeline."""

    # Constants
    CAMERA_VIEW_RANGE = 10

    def __init__(
        self,
        initial_image: np.ndarray,
        record_video: bool = False,
        video_path: str = "vo_visualization.mp4",
        fps: int = 20,
        show_info_in_video: bool = True,
        max_landmarks: int = 2000,
    ):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))
        self._max_landmarks = max_landmarks

        # --- video recording setup ---
        self._record_video = record_video
        self._video_path = video_path
        self._fps = fps
        self._video_writer: Optional[FFMpegWriter] = None
        self._show_info_in_video = show_info_in_video

        if self._record_video:
            # If ffmpeg is not installed, this will raise at runtime.
            self._video_writer = FFMpegWriter(fps=self._fps)
            # "setup" opens the file and prepares the writer
            self._video_writer.setup(self.fig, self._video_path, dpi=self.fig.dpi)

        # Create GridSpec with 3 rows, 2 columns
        # ax_image will span 2 rows (bigger), ax_keypoint_count will span 1 row (smaller)
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        ## Visualization handles
        # Image view with tracked keypoints
        self.ax_image = self.fig.add_subplot(gs[0:2, 0])
        self.img_artist = self.ax_image.imshow(initial_image, cmap="gray")
        self.kp_scatter = self.ax_image.scatter([], [], c="r", s=5)
        (self.flow_line,) = self.ax_image.plot([], [], color="y", linewidth=0.8)
        self.ax_image.set_title("Tracked Keypoints")
        self.ax_image.axis("off")

        # 3D view of landmarks and poses
        self.ax_3d = self.fig.add_subplot(gs[0:2, 1], projection="3d")
        self.landmarks_scatter = self.ax_3d.scatter([], [], [], c="b", s=1)
        (self.traj_line,) = self.ax_3d.plot([], [], [], c="g", lw=1)
        self.current_pose_scatter = self.ax_3d.scatter([], [], [], c="r", s=50)
        (self.direction_line,) = self.ax_3d.plot([], [], [], c="r", lw=2)
        self.ax_3d.set_title("3D Landmarks")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.view_init(elev=-30.0, azim=-90)

        # Keypoint count plot
        self.ax_keypoint_count = self.fig.add_subplot(gs[2, 0])
        self.keypoint_counts = []
        self.frame_indices = []
        (self.keypoint_line,) = self.ax_keypoint_count.plot([], [], c="b", linewidth=1)
        self.ax_keypoint_count.set_title("Tracked Keypoints Over Time")
        self.ax_keypoint_count.set_xlabel("Frame")
        self.ax_keypoint_count.set_ylabel("Number of Keypoints")
        self.ax_keypoint_count.grid(True, alpha=0.3)

        # 2D top-down view (X-Z plane)
        self.ax_2d = self.fig.add_subplot(gs[2, 1])
        # self.landmarks_2d_scatter = self.ax_2d.scatter([], [], c='b', s=1, alpha=0.5)
        (self.traj_2d_line,) = self.ax_2d.plot([], [], c="g", lw=2, label="Trajectory")
        self.current_pose_2d_scatter = self.ax_2d.scatter(
            [], [], c="r", s=50, label="Current Pose"
        )
        self.ax_2d.set_title("Top-Down View (X-Z)")
        self.ax_2d.set_xlabel("X")
        self.ax_2d.set_ylabel("Z")
        self.ax_2d.grid(True, alpha=0.3)
        # self.ax_2d.legend(loc='upper right')
        self.ax_2d.set_aspect("equal", adjustable="box")

        # --- info overlay (for video) ---
        # Figure-level text in lower-left; monospace + box for readability
        self.info_text = self.fig.text(
            0.01,
            0.01,
            "",
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.7, pad=3),
        )
        # Initially hidden (shown only if enabled + info string provided)
        self.info_text.set_visible(False)

    def update_image_view(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        prev_keypoints: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ):
        """Update the image and tracked keypoints."""
        self.img_artist.set_data(image)
        self.ax_image.set_title(f"Tracked Keypoints - Frame {frame_idx}")

        if keypoints.size:
            self.kp_scatter.set_offsets(keypoints)
        else:
            self.kp_scatter.set_offsets(np.empty((0, 2)))

        if prev_keypoints is not None and prev_keypoints.size:
            self._update_flow_lines(prev_keypoints, keypoints)
        else:
            self.flow_line.set_data([], [])

        # Update keypoint count plot
        self._update_keypoint_count(keypoints, frame_idx)

    def _update_flow_lines(self, prev_points: np.ndarray, curr_points: np.ndarray):
        """Update optical flow visualization lines."""
        x = np.column_stack(
            [
                prev_points[:, 0],
                curr_points[:, 0],
                np.full(prev_points.shape[0], np.nan),
            ]
        )
        y = np.column_stack(
            [
                prev_points[:, 1],
                curr_points[:, 1],
                np.full(prev_points.shape[0], np.nan),
            ]
        )
        self.flow_line.set_data(x.ravel(), y.ravel())

    def _update_keypoint_count(self, keypoints: np.ndarray, frame_idx: int):
        """Update keypoint count plot."""
        num_keypoints = keypoints.shape[0] if keypoints.size else 0
        self.keypoint_counts.append(num_keypoints)
        self.frame_indices.append(frame_idx)

        self.keypoint_line.set_data(self.frame_indices, self.keypoint_counts)

        # Auto-scale axes
        if len(self.frame_indices) > 0:
            self.ax_keypoint_count.set_xlim(0, max(self.frame_indices) + 1)
            if len(self.keypoint_counts) > 0:
                max_count = max(self.keypoint_counts)
                self.ax_keypoint_count.set_ylim(0, max_count * 1.1 + 10)

    def update_3d_view(self, landmarks: np.ndarray, poses: List[np.ndarray]):
        """Update 3D landmarks and camera trajectory."""
        if landmarks.size:
            # Limit to last N landmarks for performance
            if self._max_landmarks > 0 and landmarks.shape[0] > self._max_landmarks:
                plotting_landmarks = landmarks[-self._max_landmarks :]
            else:
                plotting_landmarks = landmarks

            self.landmarks_scatter._offsets3d = (
                plotting_landmarks[:, 0],
                plotting_landmarks[:, 1],
                plotting_landmarks[:, 2],
            )

        if len(poses) > 0:
            poses_xyz = np.array([pose[:3, 3] for pose in poses])
            self.traj_line.set_data_3d(
                poses_xyz[:, 0], poses_xyz[:, 1], poses_xyz[:, 2]
            )

            current_pose = poses[-1]
            self.current_pose_scatter._offsets3d = (
                np.array([current_pose[0, 3]]),
                np.array([current_pose[1, 3]]),
                np.array([current_pose[2, 3]]),
            )

            # Update camera-centered view
            cx, cy, cz = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
            self.ax_3d.set_xlim(
                cx - self.CAMERA_VIEW_RANGE, cx + self.CAMERA_VIEW_RANGE
            )
            self.ax_3d.set_ylim(
                cy - self.CAMERA_VIEW_RANGE, cy + self.CAMERA_VIEW_RANGE
            )
            self.ax_3d.set_zlim(
                cz - self.CAMERA_VIEW_RANGE, cz + self.CAMERA_VIEW_RANGE
            )

        # Update 2D view
        self._update_2d_view(landmarks, poses)

    def _update_2d_view(self, landmarks: np.ndarray, poses: List[np.ndarray]):
        """Update 2D top-down view (X-Z plane)."""
        # if landmarks.size:
        #     self.landmarks_2d_scatter.set_offsets(landmarks[:, [0, 2]])  # X and Z

        if len(poses) > 0:
            poses_xz = np.array([pose[[0, 2], 3] for pose in poses])
            self.traj_2d_line.set_data(poses_xz[:, 0], poses_xz[:, 1])

            current_pose = poses[-1]
            self.current_pose_2d_scatter.set_offsets(
                [[current_pose[0, 3], current_pose[2, 3]]]
            )

            # Auto-zoom to show all poses with padding
            x_min, x_max = poses_xz[:, 0].min(), poses_xz[:, 0].max()
            z_min, z_max = poses_xz[:, 1].min(), poses_xz[:, 1].max()

            # Calculate range and add 20% padding
            x_range = max(x_max - x_min, 1)  # Minimum range of 1 to avoid zero
            z_range = max(z_max - z_min, 1)
            padding_x = x_range * 0.2
            padding_z = z_range * 0.2

            self.ax_2d.set_xlim(x_min - padding_x, x_max + padding_x)
            self.ax_2d.set_ylim(z_min - padding_z, z_max + padding_z)

    def _update_info_overlay(self, info_text: Optional[str]) -> None:
        """Update (or hide) the info overlay.

        Only visible if:
        - recording is enabled, and
        - show_info_in_video was set, and
        - a non-empty string is provided.
        """
        if (
            self._record_video
            and self._show_info_in_video
            and info_text is not None
            and info_text.strip() != ""
        ):
            self.info_text.set_text(info_text)
            self.info_text.set_visible(True)
        else:
            # hide when not in use
            self.info_text.set_visible(False)

    def refresh(self):
        """Refresh the display."""
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self._record_video and self._video_writer is not None:
            # capture the current frame into the video
            self._video_writer.grab_frame()

    def step(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        prev_keypoints: Optional[np.ndarray],
        frame_idx: int,
        landmarks: np.ndarray,
        poses: List[np.ndarray],
        info_text: Optional[str] = None,
    ):
        """Update visualization for a single step."""
        self.update_image_view(image, keypoints, prev_keypoints, frame_idx)
        self.update_3d_view(landmarks, poses)
        self._update_info_overlay(info_text)
        self.refresh()

    def close(self):
        """Close visualization and show final result."""
        if self._record_video and self._video_writer is not None:
            self._video_writer.finish()  # finalize mp4
        plt.ioff()
        plt.show()
