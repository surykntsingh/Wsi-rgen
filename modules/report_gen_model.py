import torch
import torch.nn as nn

from modules.transformer import EncoderDecoder


class ReportGenModel(nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.__tokenizer = tokenizer

        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))


        self.positional_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, args.d_vf)
        )

        self.encoder_decoder = EncoderDecoder(args, tokenizer)


    def forward(self, image_embeddings, pos_embeddings, targets):
        coords_encoded = self.positional_encoder(pos_embeddings)
        patch_feats = image_embeddings + coords_encoded

        att_feats = torch.cat([self.prompt, patch_feats], dim=1)
        fc_feats = torch.sum(att_feats, dim=1)

        output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')

        return output
