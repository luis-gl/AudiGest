import librosa
import numpy as np


class AudioFeatureExtractor:
    def __init__(self,
                 audio_config: dict):
        self.fps = audio_config['fps']
        self.sample_rate = audio_config['sample_rate']
        self.sample_interval = audio_config['sample_interval']
        self.max_samples = audio_config['max_samples']
        self.window_len = audio_config['window_len']
        self.n_mfcc = audio_config['n_mfcc']
        self.n_fft = int(self.window_len * self.sample_rate)
        self.hop_length = int(self.sample_interval * self.sample_rate)

    def _get_signal_mono(self,
                         audio_path: str) -> np.ndarray:
        signal, _ = librosa.load(audio_path, sr=self.sample_rate)
        return signal

    def _check_for_padding(self,
                           mono_signal: np.ndarray) -> np.ndarray:
        signal_len = mono_signal.shape[0]
        signal_adjusted = mono_signal.copy()

        if signal_len > self.max_samples:
            signal_adjusted = mono_signal[:self.max_samples]
        elif signal_len < self.max_samples:
            pad_samples = self.max_samples - signal_len
            right_pad = (0, pad_samples)
            signal_adjusted = np.pad(mono_signal, pad_width=right_pad)

        return signal_adjusted

    def get_melspec_and_mfccs(self,
                              audio_path: str,
                              use_delta: bool = True) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        mono_signal = self._get_signal_mono(audio_path)
        signal_adjusted = self._check_for_padding(mono_signal)
        sliced_melspec = librosa.feature.melspectrogram(signal_adjusted, sr=self.sample_rate,
                                                        n_fft=self.n_fft, hop_length=self.hop_length)
        sliced_melspec = librosa.power_to_db(sliced_melspec)

        melspec = librosa.feature.melspectrogram(mono_signal, sr=self.sample_rate,
                                                 n_fft=self.n_fft, hop_length=self.hop_length)
        melspec = librosa.power_to_db(melspec)
        mfccs = librosa.feature.mfcc(S=melspec, n_mfcc=self.n_mfcc)
        if use_delta:
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs = np.concatenate((mfccs, mfccs_delta), axis=0)
        mean, std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
        mfccs = (mfccs - mean) / std

        return sliced_melspec, mfccs, melspec

    def process_framed_feature(self,
                               features: np.ndarray,
                               f_type: str,) -> np.ndarray:
        frame_size = self.fps // 2
        element_size = features.shape[0]
        pad_shape = (element_size, frame_size)
        if f_type == 'melspec':
            fill_with = features.min()
            pad_left = np.full(pad_shape, fill_value=fill_with)
            pad_right = np.full(pad_shape, fill_value=fill_with)
        else:
            pad_left = np.zeros(pad_shape)
            pad_right = np.zeros(pad_shape)
        features = np.column_stack((pad_left, features, pad_right))

        seq_len = features.shape[1]
        input_list = []
        for i in range(frame_size, seq_len - frame_size):
            input_list.append(features[:, i - frame_size:i + frame_size])
        input_list = np.array(input_list)

        return input_list
