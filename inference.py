import torch
import torch.nn.functional as F

from utils.audio import AudioFeatureExtractor
from utils.files.save import load_numpy
from utils.model.SequenceRegressor import SequenceRegressor
from config_creator import get_config
from training import get_last_epoch
# from utils.rendering.model_render import ModelRender


def get_emotion2idx_dict(config):
    emotions = config['emotions']
    emotion2idx = {}
    for idx, emotion in enumerate(emotions):
        emotion2idx[emotion] = idx

    return emotion2idx

def make_inference(audio_path, emotion, base_face_path, out_video_path):
    config = get_config()
    face = load_numpy(base_face_path)
    emotion2idx = get_emotion2idx_dict(config)
    feature_extractor = AudioFeatureExtractor(config['audio'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = SequenceRegressor(config, device)
    last_epoch = get_last_epoch()
    model.load(last_epoch)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        _, mfccs, _ = feature_extractor.get_melspec_and_mfccs(audio_path=audio_path)
        feature = torch.from_numpy(mfccs).type(torch.float32)

        template = torch.from_numpy(face).type(torch.float32)
        template = template.repeat(feature.shape[1], 1, 1)

        emotion = torch.Tensor([emotion2idx[emotion]]).type(torch.int64)

        print('emotion:', emotion.shape)
        print('feature:', feature.shape)
        print('template:', template.shape)
        print('-' * 10)

        emotion = F.one_hot(emotion, len(config['emotions']))

        feature = feature.unsqueeze(dim=0)
        feature = feature.permute(0, 2, 1)

        template = template.unsqueeze(dim=0)

        emotion = emotion.to(device)
        feature = feature.to(device)
        template = template.to(device)

        reconstructed = model(feature, emotion, None, template)
        reconstructed = reconstructed.squeeze(dim=0).cpu().numpy()
        print(reconstructed.shape)

        # renderer = ModelRender(config=config, dataset=test_data)
        # renderer.render_sequences(model, device, output_video_path)
