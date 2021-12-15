import argparse

from inference import make_inference


parser = argparse.ArgumentParser(description='Execute inference on model')
parser.add_argument('--audio_path', default='audio/MEAD_test_audio.wav', help='Path corresponding to audio with speech')
parser.add_argument('--emotion', default='happy', help='Emotion condition for emotional speech',
                    choices=['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised'])
parser.add_argument('--base_face', default='faces/base_face.npy', help='The 3D face that will be used for predicting movements for speech, the path is read from processed_data folder')
parser.add_argument('--video_directory', default='videos', help='Directory that will contain the output video')
parser.add_argument('--video_fname', default='output', help='Video file name without extension, it will generated a wmv file')

args = parser.parse_args()

audio_path = args.audio_path
emotion = args.emotion
base_face = args.base_face
video_directory = args.video_directory
video_fname = args.video_fname

make_inference(audio_path, emotion, base_face, video_directory, video_fname)