import torch
from config import *
from modules.PaddleOCR.tools.infer import utility as paddle_ultility
from modules.PaddleOCR.tools.infer import predict_det2 as ppocr_det
from modules.vietocr.vietocr.tool.config2 import Cfg
from modules.vietocr.vietocr.tool.predictor import Predictor


def models_init():
    args_paddle = paddle_ultility.parse_args()
    args_paddle.det_model_dir = paddle_model_path
    text_detector = ppocr_det.TextDetector(args_paddle)

    gpu_id = 0
    config_vietocr = Cfg.load_config_from_file(vietocr_base_cfg_path, vietocr_config_path)
    config_vietocr['weights'] = vietocr_weight_path
    config_vietocr['cnn']['pretrained'] = False
    config_vietocr['device'] = 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu'
    config_vietocr['predictor']['beamsearch'] = False
    text_recognizer = Predictor(config_vietocr)

    return text_detector, text_recognizer
