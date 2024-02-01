# Bert-VITS2

VITS2 Backbone with bert
## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应该参阅代码自己学习如何训练。
### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)


##  Usage (By ZWL)

```
pip install g2p_en
```

To use Japanese:

```
pip install mecab-python3
pip3 install unidic
python -m unidic download
```

## Usage (By Xumin)
```
python preprocess_text_esfrderu.py --transcription-path dataset/ru_dataset/ru_transcription.txt  [--path_prefix /data/]
```