import torch
from torch.utils.data import Dataset, DataLoader
from nnmnkwii.datasets import vctk
import math
import numpy as np
import os

from text import text_to_sequence
import Audio
import hparams

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class Tacotron2Dataset(Dataset):
    """ VCTK """

    def __init__(self, dataset_path=hparams.dataset_path):
        speakers = list()
        for file in os.listdir(os.path.join(hparams.VCTK_dataset_path, "wav48")):
            speakers.append(str(file[1:4]))
        self.speaker_list = speakers
        # print(speakers)
        td = vctk.TranscriptionDataSource(
            hparams.VCTK_dataset_path, speakers=speakers)
        transcriptions = td.collect_files()
        wav_paths = vctk.WavFileDataSource(
            hparams.VCTK_dataset_path, speakers=speakers).collect_files()

        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = transcriptions
        self.wav_paths = wav_paths

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        speaker_id = self.speaker_list.index(self.wav_paths[idx][31:34])
        mel_target = Audio.tools.get_mel(self.wav_paths[idx]).numpy().T

        character = self.text[idx]
        character = text_to_sequence(character, hparams.text_cleaners)
        character = np.array(character)

        stop_token = np.array([0. for _ in range(mel_target.shape[0])])
        stop_token[-1] = 1.

        sample = {"text": character,
                  "mel_target": mel_target,
                  "stop_token": stop_token,
                  "speaker_id": speaker_id}

        return sample


def _process(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    stop_tokens = [batch[ind]["stop_token"] for ind in cut_list]
    indexs = [batch[ind]["speaker_id"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_mel = np.array([])
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_normal(texts)
    stop_tokens = pad_normal(stop_tokens, PAD=1.)
    mel_targets = pad_mel(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "stop_token": stop_tokens,
           "length_mel": length_mel,
           "length_text": length_text,
           "speaker_id": indexs}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(_process(batch, cut_list[i]))

    return output


def pad_normal(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_mel(inputs):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)
                              [0]), mode='constant', constant_values=PAD)
        return x_padded[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output
