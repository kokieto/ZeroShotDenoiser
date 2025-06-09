from sklearn.utils import Bunch

cfg = Bunch()
cfg.sr = 48000
cfg.dt = 1/cfg.sr
cfg.overlap_ratio = 0.5

cfg.audio_classes = ['continuous tapping', 'motor', 'fan', 'machinery', 'traffic noise', 'conversation', 'flowing river']
cfg.audio_sts = cfg.audio_classes.copy()