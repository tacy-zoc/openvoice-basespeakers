from MyShellTTSBase.api import VITS2_API
import os
import glob
import sys

ckpt_path = sys.argv[1]
root_folder = os.path.dirname(ckpt_path)

if os.path.exists(f'{root_folder}/config_v2.json'):
    model = VITS2_API(config_path=f'{root_folder}/config_v2.json')
else:
    model = VITS2_API(config_path=f'{root_folder}/config.json')
model.load_ckpt(ckpt_path)

speaker_ids = model.hps.data.spk2id
speakers = list(speaker_ids.keys())

if 'zh' in root_folder:
    texts = open('basetts_test_resources/zh_mix_en_egs_text.txt', 'r').readlines()
    language = 'ZH_MIX_EN'
elif 'es' in root_folder:
    texts = open('basetts_test_resources/es_egs_text.txt', 'r').readlines()
    language = 'SP'
elif 'fr' in root_folder:
    texts = open('basetts_test_resources/fr_egs_text.txt', 'r').readlines()
    language = 'FR'
elif 'en' in root_folder:
    texts = open('basetts_test_resources/en_egs_text.txt', 'r').readlines()
    # texts = ["Boss? You're not my boss, you're just a sad little person who likes to hide behind a computer screen and pretend you have power over others. "]
    language = 'EN'
elif 'jp' in root_folder:
    texts = open('basetts_test_resources/jp_egs_text.txt', 'r').readlines()
    language = 'JP'
elif 'KR' in root_folder:
    texts = open('basetts_test_resources/kr_egs_text.txt', 'r').readlines()
    language = 'KR'
else:
    raise NotImplementedError()
    texts = [
            "Boss? You're not my boss, you're just a sad little person who likes to hide behind a computer screen and pretend you have power over others. ",
            "Well, in order to transport a horse to Mars, we would need to create a special spacecraft that's designed to transport live animals.",
            "Now 9 language voices and instant English voice cloning are available. With just 20 seconds of English audio, you can simulate any voice you desire. Click 'Mine' to start cloning.",
            "Welcome to our innovative chatbot platform, where you can create personalized chatbots called Shell. Discover a world of interactive conversations with My Shell, the innovative ey eye chatbot platform designed to entertain, educate, and assist you in various aspects of your life."
    ]
    language = 'EN'

save_dir = os.path.join('basetts_outputs', root_folder.split('/')[-1])

for speed in [1.0]:
    for speaker in speakers:
        for sent_id, text in enumerate(texts):
            output_path = f'{save_dir}/{speaker}/speed_{speed}/sent_{sent_id:03d}.wav'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.tts_to_file(text, speaker_ids[speaker], output_path, language=language, speed=speed)