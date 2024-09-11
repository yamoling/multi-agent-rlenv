import os
from datetime import datetime
from typing import Literal, Optional, TypeVar
from marlenv.models import ActionSpace
import numpy as np
import cv2

from .rlenv_wrapper import MARLEnv, RLEnvWrapper

A = TypeVar("A", bound=ActionSpace)
D = TypeVar("D")
S = TypeVar("S")
R = TypeVar("R", bound=float | np.ndarray)


class VideoRecorder(RLEnvWrapper[A, D, S, R]):
    """Records a video of the run"""

    FPS = 10

    def __init__(
        self,
        env: MARLEnv[A, D, S, R],
        video_folder: Optional[str] = None,
        video_encoding: Literal["mp4", "avi"] = "mp4",
    ) -> None:
        super().__init__(env)
        if video_folder is None:
            video_folder = "videos/"
        self.video_folder = video_folder
        self.video_extension = video_encoding
        self._video_count = 0
        self._recorder = None
        match video_encoding:
            case "mp4":
                self._four_cc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            case "avi":
                self._four_cc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
            case other:
                raise ValueError(f"Unsupported file video encoding: {other}")

    def step(self, actions):
        if self._recorder is None:
            raise RuntimeError("VideoRecorder not initialized")
        obs, r, done, truncated, info = super().step(actions)
        self._recorder.write(self.render("rgb_array"))
        if done:
            self._recorder.release()
        return obs, r, done, truncated, info

    def reset(self):
        res = super().reset()
        image = self.render("rgb_array")
        height, width, _ = image.shape
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_name = os.path.join(self.video_folder, f"{self._video_count}-{timestamp}.{self.video_extension}")
        os.makedirs(self.video_folder, exist_ok=True)
        self._recorder = cv2.VideoWriter(video_name, self._four_cc, VideoRecorder.FPS, (width, height))
        self._recorder.write(image)
        self._video_count += 1
        return res

    def __del__(self):
        if self._recorder is not None:
            self._recorder.release()
