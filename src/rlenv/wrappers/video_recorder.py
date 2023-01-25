from dataclasses import dataclass
from datetime import datetime
import os
import cv2
from rlenv.models import RLEnv


from .rlenv_wrapper import RLEnvWrapper


@dataclass
class VideoRecorder(RLEnvWrapper):
    """Records a video of the run"""
    video_prefix: str
    _recorder: cv2.VideoWriter

    def __init__(self, env: RLEnv, video_folder: str = None) -> None:
        super().__init__(env)
        if not video_folder:
            video_folder = "videos/"
        directory = os.path.dirname(video_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.video_prefix = video_folder
        self._video_count = 0
        self._recorder = None

    def step(self, actions):
        res = super().step(actions)
        self._recorder.write(self.render("rgb_array"))
        return res

    def reset(self):
        res = super().reset()
        image = self.render("rgb_array")
        if self._recorder is not None:
            self._recorder.release()
        height, width, _ = image.shape
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_name = f"{self._video_count}-{self.video_prefix}{timestamp}.avi"
        fps = 10
        self._recorder = cv2.VideoWriter(video_name, 0, fps, (width, height))
        self._recorder.write(image)
        self._video_count += 1
        return res

    def __del__(self):
        if self._recorder is not None:
            self._recorder.release()
