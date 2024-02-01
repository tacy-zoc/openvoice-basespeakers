# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import sys
sys.path.append("./")
from transformers import AutoTokenizer
from . import punctuation, symbols
from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer
from .cleaner_multiling import unicleaners


ph_de = None
def german_text_to_phonemes(text) -> str:
    global ph_de
    if ph_de is None:
        ph_de = MultiPhonemizer({"de": "espeak"})
    phoneme = ph_de.phonemize(text, separator="", language='de')
    return phoneme

def text_normalize(text):
    text = unicleaners(text, cased=True, lang='de')
    return text

def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

model_id = 'bert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)

        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", german_text_to_phonemes(w)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph =  [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda'):
    from text import german_bert
    return german_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    from text.symbols import symbols
    import json
    from tqdm import tqdm

    lines = open('./text/de_text.txt').readlines()
    new_symbols = []

    for i, line in tqdm(enumerate(lines)):
        text = line.strip()
        text = text_normalize(text)
        phones, tones, word2ph = g2p(text)
        if i % 2000 == 0:
            print(''.join(phones))
        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols.append(ph)
                print('update!, now symbols: \n')
                print(new_symbols)
                print(''.join(phones))
                data = {
                    'symbols': new_symbols
                }
                with open('./text/de_symbols.json', 'w') as f:
                    json.dump(data, f, indent=4)