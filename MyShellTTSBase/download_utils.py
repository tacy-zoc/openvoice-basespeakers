import torch
import os
from . import utils

DOWNLOAD_CKPT_URLS = {
    'EN': 'https://cloud.tsinghua.edu.cn/f/0ede98651942498e8746/?dl=1',
    'EN_V2': 'https://cloud.tsinghua.edu.cn/f/7df6942350964f839037/?dl=1',
    'FR': 'https://cloud.tsinghua.edu.cn/f/d8750fe6db0140d19866/?dl=1',
    'JP': 'https://cloud.tsinghua.edu.cn/f/b28e323511ae4e2899ae/?dl=1',
    'ES': 'https://cloud.tsinghua.edu.cn/f/47d96692dd6a4be0b3e8/?dl=1',
    'ZH': 'https://cloud.tsinghua.edu.cn/f/e08c061df7e04d47bcc7/?dl=1',
    'KR': 'https://cloud.tsinghua.edu.cn/f/4dab240a50994709a714/?dl=1',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://cloud.tsinghua.edu.cn/f/b3240c1b95e74d38962e/?dl=1',
    'EN_V2': 'https://cloud.tsinghua.edu.cn/f/4622c302429249fb87b3/?dl=1',
    'FR': 'https://cloud.tsinghua.edu.cn/f/4e376d97f13d4b12b0ee/?dl=1',
    'JP': 'https://cloud.tsinghua.edu.cn/f/d5538c744d344401980d/?dl=1',
    'ES': 'https://cloud.tsinghua.edu.cn/f/3a839e56e14c4f918d16/?dl=1',
    'ZH': 'https://cloud.tsinghua.edu.cn/f/49e33f0571084bdcbb5c/?dl=1',
    'KR': 'https://cloud.tsinghua.edu.cn/f/e6834736f96946638821/?dl=1',
}

def load_or_download_config(locale):
    language = locale.split('-')[0].upper()
    assert language in DOWNLOAD_CONFIG_URLS
    config_path = os.path.expanduser(f'~/.local/share/openvoice/basespeakers/{language}/config.json')
    try:
        return utils.get_hparams_from_file(config_path)
    except:
        # download
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.system(f'wget {DOWNLOAD_CONFIG_URLS[language]} -O {config_path}')
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device):
    language = locale.split('-')[0].upper()
    assert language in DOWNLOAD_CKPT_URLS
    ckpt_path = os.path.expanduser(f'~/.local/share/openvoice/basespeakers/{language}/checkpoint.pth')
    try:
        return torch.load(ckpt_path, map_location=device)
    except:
        # download
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        os.system(f'wget {DOWNLOAD_CKPT_URLS[language]} -O {ckpt_path}')
    return torch.load(ckpt_path, map_location=device)