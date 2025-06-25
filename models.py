import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore

from modules.loss import LanguageModelCriterion
from modules.report_gen_model import ReportGenModel


class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer, lr=5e-5, weight_decay=0.01):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)
        self.tokenizer = tokenizer
        self.__lr = lr
        self.__weight_decay = weight_decay
        self.rouge = ROUGEScore()
        self.bleu = BLEUScore(n_gram=4)

    def loss_fn(self, output, reports_ids, reports_masks):
        criterion = LanguageModelCriterion()
        loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        return loss

    def training_step(self, batch):
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch
        output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch
        output = self.model(patch_feats, pos_feats, report_ids, mode='sample')
        output_ = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        loss = self.loss_fn(output_, report_ids, report_masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
        target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())

        print(f'pred_texts: {pred_texts}, target_texts: {target_texts}')

        rouge_score = self.rouge(pred_texts, target_texts)
        bleu_score = self.bleu(pred_texts, target_texts)

        self.log('val_rouge', rouge_score['rouge1_fmeasure'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_bleu', bleu_score, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.__lr, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]