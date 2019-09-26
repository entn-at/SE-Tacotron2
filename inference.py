import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

import waveglow
import glow
from network import SVTacotron2
import speakerencoder
from text import text_to_sequence
import hparams as hp
import Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')

    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_model(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(SVTacotron2(hp)).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path))['model'])
    model.eval()

    return model


def synthesis(model, text, speaker_id):
    with torch.no_grad():
        sequence = np.array(text_to_sequence(
            text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.module.inference(
            sequence, speaker_id)

        return mel_outputs[0].cpu(), mel_outputs_postnet[0].cpu(), mel_outputs_postnet


if __name__ == "__main__":
    # Test
    num = 6
    text = "Hi, how are you."
    speaker_id = torch.Tensor([num]).unsqueeze(0).long().to(device)

    model = get_model(178000)
    mel, mel_postnet, mel_postnet_torch = synthesis(model, text, speaker_id)
    if not os.path.exists("results"):
        os.mkdir("results")
    Audio.tools.inv_mel_spec(
        mel_postnet, os.path.join("results", text + str(num) + "griffin_lim.wav"))
    plot_data([mel.numpy(), mel_postnet.numpy()])

    waveglow_path = os.path.join("waveglow", "pre_trained_model")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.inference.inference(
        mel_postnet_torch, wave_glow, os.path.join("results", text + str(num) + "waveglow.wav"))
