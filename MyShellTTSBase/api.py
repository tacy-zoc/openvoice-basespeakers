import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv

class VITS2_API(nn.Module):
    def __init__(self, 
                config_path, 
                device='cuda:0'):
        super().__init__()
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        if getattr(hps, 'num_languages', None) is None:
            from text.symbols import symbols
            from text.symbols import num_languages
            from text.symbols import num_tones
            cfg = json.load(open(config_path))
            cfg['symbols'] = symbols
            cfg['num_tones'] = num_tones
            cfg['num_languages'] = num_languages
            with open(config_path.replace('.json', '_v2.json'), 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            print('save v2 json for this model ... ')
        else:
            print('Load symbols from config for this model ... ')
            num_languages = hps.num_languages
            num_tones = hps.num_tones
            symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device


    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)

        iteration = checkpoint_dict.get('iteration', 0)
        print("Loaded checkpoint '{}' (iteration {})".format(ckpt_path, iteration))

    def _get_se(self, ref_wav_list, se_save_path=None):
        device = self.device
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        elif isinstance(ref_wav_list, np.ndarray):
            gs = torch.from_numpy(ref_wav_list)
            if se_save_path is not None:
                os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
                torch.save(gs, se_save_path)
            return gs.to(device)
        elif isinstance(ref_wav_list, torch.Tensor):
            gs = ref_wav_list
            if se_save_path is not None:
                os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
                torch.save(gs, se_save_path)
            return gs.to(device)

        hps = self.hps
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                        center=False).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs, se_save_path)

        return gs

    def voice_conversion(self, audio_src_path, src_ref_wav_list, tgt_ref_wav_list, output_path=None, tau=0.3):
        hps = self.hps

        # load audio
        print(f'####  note that vc use 44100 Hz audio. ####')
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        print(f'input audio sample_rate is {sample_rate}')
        audio = torch.tensor(audio).float()

        # get se
        sid_src = self._get_se(src_ref_wav_list)
        sid_tgt = self._get_se(tgt_ref_wav_list) # 1 256 1
        # print(sid_src.shape, sid_tgt.shape)
        
        with torch.no_grad():
            y = torch.FloatTensor(audio).cuda()
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False).cuda()
            spec_lengths = torch.LongTensor([spec.size(-1)]).cuda()
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt, tau=tau)[0][
                        0, 0].data.cpu().float()

            if output_path is None:
                return audio.numpy()
            else:
                torchaudio.save(output_path, audio.unsqueeze(0), 44100)

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05)/speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language):
        texts = split_sentence(text, language_str=language)
        print(" > Text splitted to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, language='EN', force_disable_bert=False):
        texts = self.split_sentences_into_pieces(text, language)
        audio_list = []
        for t in texts:
            if language == 'EN':
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id, force_disable_bert=force_disable_bert)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)

    def get_vc_onnx_forward_fn(self, audio, sid_src, sid_tgt, tau):
        hps = self.hps
        with torch.no_grad():
            y = audio
            spec = spectrogram_torch_conv(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(y.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt, tau=tau)[0][0, 0]
        return audio

    def convert_vc_model_into_onnx(self, onnx_model_path, src_ref_wav_list, tgt_ref_wav_list, tau=0.3):
        self.model.eval()
        # define a dict and its names
        dummy_inputs = []
        dummy_inputs_name = []

        # swap the forward func
        self.forward = self.get_vc_onnx_forward_fn

        # append other input and value
        
        seg_length = 44100 * 20  # Example seg_length
        dummy_audio = torch.randn(1, seg_length).cuda()  # Example input 'z'
        dummy_inputs.append(dummy_audio)
        dummy_inputs_name.append('audio')

        sid_src = self._get_se(src_ref_wav_list).float()
        dummy_inputs.append(sid_src)
        dummy_inputs_name.append('sid_src')

        sid_tgt = self._get_se(tgt_ref_wav_list).float()
        dummy_inputs.append(sid_tgt)
        dummy_inputs_name.append('sid_tgt')

        tau = torch.FloatTensor([tau]).cuda()
        dummy_inputs.append(tau)
        dummy_inputs_name.append('tau')


        # Export the PyTorch model to ONNX
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            onnx_model_path,
            opset_version=15,  # Specify the desired ONNX opset version
            input_names=dummy_inputs_name,  # Specify input names
            dynamic_axes={"audio": {1: "seg_length"}},  # Specify dynamic axes for 'audio'
            output_names=["output"],  # Specify output names
        )

        # trace back
        # self.forward = _forward

    def get_tts_onnx_forward_fn_head(self, x, x_lengths, sid, tone, language, bert, ja_bert, sdp_ratio):
        noise_scale_w = 0.8
        
        g = self.model.emb_g(sid).unsqueeze(-1)  # [b, h, 1] 
        g_p = g

        x, m_p, logs_p, x_mask = self.model.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        logw = self.model.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.model.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask

        return w, x_mask, m_p, logs_p

    def get_tts_onnx_forward_fn_body(self, w, x_mask, m_p, logs_p, length_scale):
        # length_scale = 1. / speed
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        return m_p, logs_p, y_mask, y_lengths

    def get_tts_onnx_forward_fn_tail(self, m_p, logs_p, y_mask, sid):
        noise_scale = 0.6

        g = self.model.emb_g(sid).unsqueeze(-1) # [b, h, 1]

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.model.flow(z_p, y_mask, g=g, reverse=True)
        o = self.model.dec((z * y_mask)[:,:,:None], g=g)
        return o[0, 0]

    def convert_tts_model_into_onnx_split(self, onnx_model_prefix='vits_base_op15'):
        """Export model to ONNX format for inference

        Args:
            output_path (str): Path to save the exported model.
            verbose (bool): Print verbose information. Defaults to True.
        """

        # set export mode
        self.disc = None
        self.model.eval()

        self.forward = self.get_tts_onnx_forward_fn_head

        ################################  head ###############################
        # set dummy inputs
        dummy_input_length = 100
        sequences = torch.randint(low=0, high=100, size=(1, dummy_input_length), dtype=torch.long).cuda()
        sequence_lengths = torch.LongTensor([sequences.size(1)]).cuda()
        sid =  torch.LongTensor([0]).cuda()
        tones = torch.randint(low=0, high=1, size=(1, dummy_input_length), dtype=torch.long).cuda()
        lang_ids = torch.randint(low=0, high=2, size=(1, dummy_input_length), dtype=torch.long).cuda()
        bert = torch.rand(1, 1024, dummy_input_length).cuda()
        ja_bert = torch.rand(1, 768, dummy_input_length).cuda()
        sdp_ratio = torch.FloatTensor([0.2]).cuda()

        
        dummy_input = (sequences, sequence_lengths, sid, tones, lang_ids, bert, ja_bert, sdp_ratio)
        
        # export to ONNX
        torch.onnx.export(
            model=self,
            args=dummy_input,
            f=f"{onnx_model_prefix}_head.onnx",
            opset_version=15,
            input_names=["input", "input_lengths", "sid", "tones", "lang_ids", "bert", "ja_bert",  "sdp_ratio"],
            output_names=["w", "x_mask", "m_p", "logs_p"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "sid": {0: "batch_size"},
                "tones": {0: "batch_size", 1: "phonemes"},
                "lang_ids": {0: "batch_size", 1: "phonemes"},
                "bert": {0: "batch_size", 2: "phonemes"},
                "ja_bert": {0: "batch_size", 2: "phonemes"},
            },
        )

        head_res = self.get_tts_onnx_forward_fn_head(sequences, sequence_lengths, sid, tones, lang_ids, bert, ja_bert, sdp_ratio)
        print('output shape of head', head_res[0].shape, head_res[1].shape, head_res[2].shape, head_res[3].shape)
        body_res = self.get_tts_onnx_forward_fn_body(head_res[0], head_res[1], head_res[2], head_res[3], length_scale=1.)
        print('output shape of body', body_res[0].shape, body_res[1].shape, body_res[2].shape, body_res[3].shape)


        dummy_input_length = 200
        sequences = torch.randint(low=0, high=100, size=(1, dummy_input_length), dtype=torch.long).cuda()
        sequence_lengths = torch.LongTensor([sequences.size(1)]).cuda()
        sid =  torch.LongTensor([0]).cuda()
        tones = torch.randint(low=0, high=1, size=(1, dummy_input_length), dtype=torch.long).cuda()
        lang_ids = torch.randint(low=0, high=2, size=(1, dummy_input_length), dtype=torch.long).cuda()
        bert = torch.rand(1, 1024, dummy_input_length).cuda()
        ja_bert = torch.rand(1, 768, dummy_input_length).cuda()
        sdp_ratio = torch.FloatTensor([0.2]).cuda()
        
        dummy_input = (sequences, sequence_lengths, sid, tones, lang_ids, bert, ja_bert, sdp_ratio)

        head_res = self.get_tts_onnx_forward_fn_head(sequences, sequence_lengths, sid, tones, lang_ids, bert, ja_bert, sdp_ratio)
        print('output shape of head', head_res[0].shape, head_res[1].shape, head_res[2].shape, head_res[3].shape)
        body_res = self.get_tts_onnx_forward_fn_body(head_res[0], head_res[1], head_res[2], head_res[3], length_scale=1.)
        print('output shape of body', body_res[0].shape, body_res[1].shape, body_res[2].shape, body_res[3].shape)

        # exit()
        ################################  tail ###############################
        self.forward = self.get_tts_onnx_forward_fn_tail

        # set dummy inputs
        dummy_input_length = 300
        m_p = torch.randn(1, 192, dummy_input_length).cuda()
        logs_p = torch.rand(1, 192, dummy_input_length).cuda()
        y_mask = torch.ones(1, 1, dummy_input_length).cuda()
        sid =  torch.LongTensor([0]).cuda()
        dummy_input = (m_p, logs_p, y_mask, sid)


        # export to ONNX
        torch.onnx.export(
            model=self,
            args=dummy_input,
            f=f"{onnx_model_prefix}_tail.onnx",
            opset_version=15,
            input_names=["m_p", "logs_p", "y_mask", "sid"],
            output_names=["output"],
            dynamic_axes={
                "m_p": {0: "batch_size", 2: "phonemes"},
                "logs_p": {0: "batch_size", 2: "phonemes"},
                "y_mask": {0: "batch_size", 2: "phonemes"},
                "sid": {0: "batch_size"}
            },
        )

        tail_res = self.get_tts_onnx_forward_fn_tail(m_p, logs_p, y_mask, sid)
        print('output shape of head', tail_res.shape)

        dummy_input_length = 600
        m_p = torch.randn(1, 192, dummy_input_length).cuda()
        logs_p = torch.rand(1, 192, dummy_input_length).cuda()
        y_mask = torch.ones(1, 1, dummy_input_length).cuda()
        sid =  torch.LongTensor([0]).cuda()
        dummy_input = (m_p, logs_p, y_mask, sid)
        
        tail_res = self.get_tts_onnx_forward_fn_tail(m_p, logs_p, y_mask, sid)
        print('output shape of head', tail_res.shape)