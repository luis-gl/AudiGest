import argparse

from inference import make_inference


parser = argparse.ArgumentParser(description='Execute inference on model')
parser.add_argument('--audio_path', default='audio/MEAD_test_audio.wav', help='Path corresponding to audio with speech')
parser.add_argument('--emotion', default='happy', help='Emotion condition for emotional speech',
                    choices=['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised'])
parser.add_argument('--base_face', default='faces/base_face.npy', help='The 3D face that will be used for predicting movements for speech, the path is read from processed_data folder')
parser.add_argument('--out_video_path', default='videos/output.mp4', help='Path corresponding to the output video')

args = parser.parse_args()

audio_path = args.audio_path
emotion = args.emotion
base_face = args.base_face
out_video_path = args.out_video_path

make_inference(audio_path, emotion, base_face, out_video_path)