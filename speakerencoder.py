import torch
import torch.nn as nn
import os

import hparams as hp


class SpeakerEncoder(nn.Module):
    """ Speaker Encoder """

    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        self.lstm = nn.LSTM(hp.n_mels_channel,
                            hp.hidden_dim,
                            num_layers=hp.num_layer,
                            batch_first=True)
        self.projection = nn.Linear(hp.hidden_dim, hp.speaker_dim)
        self.init_params()

    def init_params(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, input_lengths):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        out = list()
        for i in range(x.size(0)):
            out.append(x[i][input_lengths[i]-1])
        out = torch.stack(out)
        out = self.projection(out)
        out = out / torch.norm(out)

        return out


def get_model(num):
    model = SpeakerEncoder()
    model.eval()

    checkpoint = torch.load(os.path.join(
        hp.se_checkpoint_path, 'checkpoint_SE_' + str(num)+'.pth.tar'))
    model.load_state_dict(checkpoint['model'])

    return model


def get_embedding(model, mels, lengths):
    with torch.no_grad():
        speaker_embeddings = model(mels, lengths)

    return speaker_embeddings
