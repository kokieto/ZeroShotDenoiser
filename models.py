from msclap import CLAP
from config import cfg
import torch
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime
import os
from utils import write_as_wav
import torch.nn.functional as F

class AudioReconstructor():
    def __init__(self, query_temp=2.5, key_temp=1e-2, n_iter_similarity=5):
        self.key_temp = key_temp
        self.query_temp = query_temp
        self.clap = CLAP(version='2023')
        self.prompt_prefix = 'this is the sound of '
        self.n_iter = n_iter_similarity
        self.y = [self.prompt_prefix + x for x in cfg.audio_sts]
        self.audio_classes = cfg.audio_classes

    def reconstruct_audio(self, separated_amples, tgt_class):
        similarity = self.calc_similarity(separated_amples, return_type='tensor')
        y_preds = F.softmax(F.softmax(similarity/self.query_temp, dim=1)/self.key_temp, dim=0).numpy()
        att_scores_df = pd.DataFrame(y_preds).round(3)
        att_scores_df.columns = self.audio_classes
        pred_amples = np.array([score*amples for score, amples in zip(att_scores_df[tgt_class].values, separated_amples)]).sum(axis=0)
        return pred_amples, att_scores_df
    
    def calc_similarity(self, separated_amples, return_type='tensor'):
        sound_files = self.store_audio(separated_amples)
        similarities = []
        for _ in range(self.n_iter):
            text_embeddings = self.clap.get_text_embeddings(self.y)
            audio_embeddings = self.clap.get_audio_embeddings(sound_files, resample=True)
            similarity = self.clap.compute_similarity(audio_embeddings, text_embeddings).detach().cpu()
            similarities.append(similarity)
        similarity = torch.tensor(np.array(similarities)).mean(axis=0)
        return similarity if return_type == 'tensor' else similarity.numpy()
    
    def store_audio(self, separated_amples):
        cd = datetime.now()
        prefix = f'DataChache/{cd.year:02}{cd.month:02}{cd.day:02}'
        file_nm = f'{prefix}/{cd.hour:02}{cd.minute:02}{cd.second:02}_{cd.microsecond:06}_id'
        os.makedirs(prefix, exist_ok=True)
        for idx, amples in enumerate(separated_amples):
            write_as_wav(amples, f'{file_nm}{idx}')
        sound_files = [f'{file_nm}{idx}.wav' for idx in range(idx+1)]
        return sound_files
    

class PesudoSNRGenerator(AudioReconstructor):
    def __init__(self):
        super().__init__()
        
    def estimate_snr(self, pred_amples, tgt_class):
        similarity = self.calc_similarity([pred_amples])
        snr = F.softmax(similarity/(2*self.query_temp), dim=1).detach().cpu().numpy()
        snr_df = pd.DataFrame(snr).round(3)
        snr_df.columns = self.audio_classes
        return snr_df[tgt_class][0]
