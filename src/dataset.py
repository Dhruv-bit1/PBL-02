import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        root_dir structure:
            root_dir/
                fight/
                non-fight/
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

        # Class folders (fight, non-fight)
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            class_folder = os.path.join(root_dir, cls)

            for video_file in os.listdir(class_folder):
                video_path = os.path.join(class_folder, video_file)

                if video_file.endswith((".avi", ".mp4", ".mov")):
                    self.video_paths.append(video_path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros(self.num_frames, 3, 224, 224)

        frame_indices = np.linspace(
            0, total_frames - 1, self.num_frames, dtype=int
        )

        frames = []
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))

                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0

                frames.append(frame)

            current_frame += 1

        cap.release()

        # Pad if video shorter
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path)

        return frames, torch.tensor(label)
