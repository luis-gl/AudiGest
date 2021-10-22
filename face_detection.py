import cv2
import os
import numpy as np
import mediapipe as mp
from utils.files.save import save_numpy


def rescale_landmark_sequence(lmk_sequence: np.ndarray) -> np.ndarray:
    sequence = lmk_sequence.copy()
    sequence[:, :, 0] *= 16.0
    sequence[:, :, 1] *= 9.0
    sequence[:, :, 2] *= 16.0
    return sequence

class FaceDetector:
    def __init__(self,
                 confidence: float = 0.5,
                 apply_rescaling: bool = False,
                 display_landmarks: bool = False,
                 debug_color: tuple[int, int, int] = (255, 0, 0),
                 debug_thickness: int = 1,
                 debug_circle_radius: int = 1):
        """
        Class to extract face landmarks from video.

        Args:
            confidence: Float number that represents minimum confidence ([0.0, 0.1]) for face landmark extractor to
                be considered successful.
            apply_rescaling: Boolean that defines if scale landmark point from normalized screen space (the rescaling
                uses a 16:9 screen from MEAD videos).
            display_landmarks: Boolean that defines if show each frame with face landmarks.
            debug_color: Color tuple of integers (r, g, b) that will be applied to the landmark drawer when
                displayed.
            debug_thickness: Integer number that represents thickness that will be applied to the landmark drawer
                when displayed.
            debug_circle_radius: Integer number that represents radius of the points drawn by landmark displayer.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=debug_color, thickness=debug_thickness,
                                                        circle_radius=debug_circle_radius)
        self.min_detection_confidence = confidence
        self.min_tracking_confidence = confidence
        self.display_landmarks = display_landmarks
        self.apply_rescaling = apply_rescaling

    def _rescale_landmark(self, landmark_point: list) -> list:
        landmark_point[0] *= 16.0
        landmark_point[1] *= 9.0
        landmark_point[2] *= 16.0
        return landmark_point

    def get_face_landmarks(self, video_path: str = "") -> np.ndarray:
        """
        Process a video and returns an array of named tuples of the face landmarks detected.

        Args:
            video_path: String containing the path of the video file.

        Raises:
            ValueError: If the video path is not a video file or does not exists.

        Returns:
            A list of lists containing the x, y and z component of the face landmarks detected on each video frame.
        """
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            raise ValueError('Video path is not a file or does not exists.')

        if not video_path.endswith('.mp4'):
            raise ValueError('Video path is not a supported video')

        video = cv2.VideoCapture(video_path)
        video_landmarks = []
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence) as face_mesh:
            while video.isOpened():
                success, img = video.read()
                if not success:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = face_mesh.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if not results.multi_face_landmarks:
                    continue

                frame_landmarks = results.multi_face_landmarks[0]
                lm_list = []
                for lm in frame_landmarks.landmark:
                    lm_point = [lm.x - 0.5, -(lm.y - 0.5), lm.z]
                    if self.apply_rescaling:
                        lm_point = self._rescale_landmark(lm_point)
                    lm_list.append(lm_point)
                video_landmarks.append(lm_list)

                if not self.display_landmarks:
                    continue

                self.mp_drawing.draw_landmarks(image=img, landmark_list=frame_landmarks,
                                               # connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.drawing_spec,
                                               connection_drawing_spec=self.drawing_spec)
                cv2.imshow('Face landmarks', img)
                if cv2.waitKey(30) == ord('s'):
                    break

        video.release()
        cv2.destroyAllWindows()

        video_landmarks = np.array(video_landmarks)
        return video_landmarks


def main():
    video_path = 'MEAD/M003/video/front/angry/level_2/001.mp4'
    detector = FaceDetector(display_landmarks=True, confidence=0.8)
    video_landmarks = detector.get_face_landmarks(video_path=video_path)
    print(video_landmarks.shape)
    # save_numpy(video_landmarks, 'test.npy')


if __name__ == "__main__":
    main()
