import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
from typing import List
from math import sqrt
import logging
import os
from tacotron2.layers import ConvNorm, LinearNorm, PositionalEncoding, LocationLayer, ACTIVATION_GROUP
import hydra
from torch.nn import functional as F
from tacotron2.utils import get_abspath
import numpy as np
import math
from torch.autograd import Variable

class Tacotron2(nn.Module):
    """
        Tacotron2 module:
            - Encoder
            - Decoder
            - Postnet
    """

    model_save_name = 'pretrained_model.bin'

    def __init__(self, cfg):
        super(Tacotron2, self).__init__()

        """
            Embedding with uniform init
            https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/model.py#L464
        """
        self.embedding = nn.Embedding(
            cfg.num_labels, cfg.encoder_embedding_dim)
        std = sqrt(2.0 / (cfg.num_labels + cfg.encoder_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        ## Encoder
        self.encoder = Encoder(cfg)

        ## decoder
        self.decoder = Decoder(cfg)

        ## postnet
        self.postnet = Postnet(cfg)

        self.cfg = cfg
        self.sampling_rate = cfg.sampling_rate
        self.loss_masking = cfg.loss_masking

        ## loss params
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        logging.info('build tacotron2 model')


    def forward(self, text_ids, text_lengths, mel_specs, mel_lengths, **kwargs):
        """
            :param text_ids: token ids corresponding to input sentences [B, N]
            :param text_lengths: length of text_ids [B]
            :param mel_specs: log mel-spectrogram extracted from raw audio [B, n_mel_channels, T]
            :param kwargs: extra
        """

        
        ## embedded_inputs :  [B, N, encoder_embedding_dim]
        embedded_inputs = self.embedding(text_ids)
        
        ## encoder_outputs  : [B, N, encoder_lstm_dim]
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specs, text_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs) # [B, n_mel_channels, T]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return {
            'mel_outputs' : mel_outputs,
            'mel_outputs_postnet' : mel_outputs_postnet,
            'gate_outputs' : gate_outputs,
            'alignments' : alignments
        }

    def inference(self, text_ids, text_lengths=None, **kwargs):
        embedded_inputs = self.embedding(text_ids)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs) # [B, n_mel_channels, T]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return {
            'mel_outputs' : mel_outputs,
            'mel_outputs_postnet' : mel_outputs_postnet,
            'gate_outputs' : gate_outputs,
            'alignments' : alignments
        }

    def get_loss(self,
                 mel_outputs, mel_outputs_postnet, gate_outputs, ## output
                 mel_specs, mel_lengths, gate_targets, ## input
                 **kwargs):
        """
            total loss are sum of
            1. mse loss of pre-defined mel-spectrogram
            2. mae loss of pre-defined mel-spectrogram
            3. mse loss of post-processed(postnet) mel-spectrogram
            4. mae loss of post-processed(postnet) mel-spectrogram
            5. bce loss of gateway witch indicate where to finish
        """

        ## generate mel mask : [B, n_mel_channels, T]
        B = mel_outputs.size(0)
        N = mel_outputs.size(1) ## n_mel_channels
        max_len = mel_lengths.max()
        ids = torch.arange(max_len, device=mel_lengths.device).expand(B, N, max_len)
        mel_mask = ~(ids < mel_lengths.view(B, 1, 1)).transpose(1, 2).bool()
        
        ## 1. + 2. mel
        pred = mel_outputs.transpose(1, 2)
        post_pred = mel_outputs_postnet.transpose(1, 2)
        label = mel_specs.transpose(1, 2)
        if self.loss_masking:
            pred.data.masked_fill_(mel_mask, 0.0)
            post_pred.data.masked_fill_(mel_mask, 0.0)
        
        predicted_mse = self.mse(pred, label)
        predicted_mse = torch.nan_to_num(predicted_mse)

        predicted_mae = self.mae(pred, label)
        predicted_mae = torch.nan_to_num(predicted_mae)

        ## 3. + 4. postnet mel
        predicted_postnet_mse = self.mse(post_pred, label)
        predicted_postnet_mse = torch.nan_to_num(predicted_postnet_mse)

        predicted_postnet_mae = self.mae(post_pred, label)
        predicted_postnet_mae = torch.nan_to_num(predicted_postnet_mae)

        ## 5. gate
        g_ids = torch.arange(max_len, device=mel_lengths.device).expand(B, max_len)
        gate_mask = ~(g_ids < mel_lengths.view(B, 1)).bool()
        gate_pred = gate_outputs
        if self.loss_masking:
            gate_pred.data.masked_fill_(gate_mask, 1e3)
            
        
        predicted_gate_bce = self.bce(gate_outputs, gate_targets)
        predicted_gate_bce = torch.nan_to_num(predicted_gate_bce)

        loss = predicted_mse + predicted_mae + predicted_postnet_mse + predicted_postnet_mae + predicted_gate_bce
        return { 
            'loss':loss,
            'predicted_mse':predicted_mse,
            'predicted_mae':predicted_mae,
            'predicted_postnet_mse':predicted_postnet_mse,
            'predicted_postnet_mae':predicted_postnet_mae,
            'predicted_gate_bce':predicted_gate_bce,
        }


    @classmethod
    def from_pretrained(cls, pretrained_path):
        pretrained_path = get_abspath(pretrained_path)
        logging.info('load files from [{}]'.format(pretrained_path))
        state = torch.load(os.path.join(pretrained_path, cls.model_save_name))
        cfg = state['cfg']
        model = cls(cfg)
        model.load_state_dict(state['model'])

        return model

    def save_pretrained(self, save_path):
        save_path = get_abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        state = {
            'cfg' : self.cfg,
            'model' : self.state_dict(),
        }
        torch.save(state, os.path.join(save_path, self.model_save_name))
        logging.info('save files to [{}]'.format(save_path))



class Encoder(nn.Module):
    """Encoder module:
        - Three (dropout, batch normalization, convolution)
        - single Bidirectional LSTM
    """
    def __init__(self, cfg):
        super(Encoder, self).__init__()


        assert cfg.encoder_activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(", ".join(ACTIVATION_GROUP.keys()))

        convolutions = []
        for _ in range(cfg.encoder_n_convolutions):

            conv_layer = nn.Sequential(
                ConvNorm(cfg.encoder_embedding_dim,
                         cfg.encoder_embedding_dim,
                         kernel_size=cfg.encoder_kernel_size, stride=1,
                         padding=int((cfg.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain=cfg.encoder_activation),
                nn.BatchNorm1d(cfg.encoder_embedding_dim, momentum=1-cfg.encoder_batch_norm_decay))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)
        self.dropout = nn.Dropout(p=cfg.encoder_dropout_p)

        self.lstm = nn.LSTM(input_size=cfg.encoder_embedding_dim,
                            hidden_size=int(cfg.encoder_lstm_dim/2), num_layers=1,
                            batch_first=True, bidirectional=True)

        self.activation = ACTIVATION_GROUP[cfg.encoder_activation]()
        #torch.nn.init.xavier_uniform_(self.lstm.weight)

    def forward(self, x, input_lengths=None):
        """
            :param x: [B, N, encoder_embedding_dim]
            :param input_lengths: [B]
            :return: [B, N, encoder_lstm_dim]
        """
        input_lengths = input_lengths.cpu().tolist() if type(input_lengths) == torch.Tensor else input_lengths

        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = self.dropout(self.activation(conv(x)))

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        if input_lengths is not None:
            x = pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout_p=0.5, activation='relu'):
        super(Prenet, self).__init__()


        assert activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(
            ", ".join(ACTIVATION_GROUP.keys()))

        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = ACTIVATION_GROUP[activation]()

    def forward(self, x):
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, cfg):
        super(Postnet, self).__init__()

        assert cfg.postnet_activation in ACTIVATION_GROUP, 'activation must either one of them [{}]'.format(
            ", ".join(ACTIVATION_GROUP.keys()))

        self.activation = ACTIVATION_GROUP[cfg.postnet_activation]()
        self.dropout = nn.Dropout(p=cfg.postnet_dropout_p)

        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(cfg.n_mel_channels, cfg.postnet_embedding_dim,
                         kernel_size=cfg.postnet_kernel_size, stride=1,
                         padding=int((cfg.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain=cfg.postnet_activation),
                nn.BatchNorm1d(cfg.postnet_embedding_dim))
        )

        for i in range(1, cfg.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(cfg.postnet_embedding_dim,
                             cfg.postnet_embedding_dim,
                             kernel_size=cfg.postnet_kernel_size, stride=1,
                             padding=int((cfg.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain=cfg.postnet_activation),
                    nn.BatchNorm1d(cfg.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(cfg.postnet_embedding_dim, cfg.n_mel_channels,
                         kernel_size=cfg.postnet_kernel_size, stride=1,
                         padding=int((cfg.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(cfg.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.dropout(self.activation(self.convolutions[i](x)))
        x = self.dropout(self.convolutions[-1](x))

        return x



class Attention(nn.Module):
    def __init__(self, attention_lstm_dim, encoder_lstm_dim, attention_dim,
                 attention_n_filters, attention_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_lstm_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(encoder_lstm_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_n_filters,
                                            attention_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights




class Decoder(nn.Module):
    """Decoder module:
        - Three (dropout, batch normalization, convolution)
        - single Bidirectional LSTM
    """

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        ## decoder

        self.n_mel_channels = cfg.n_mel_channels
        self.decoder_lstm_dim = cfg.decoder_lstm_dim
        self.n_frames_per_step = cfg.n_frames_per_step

        self.decoder_lstm_dim = cfg.decoder_lstm_dim
        self.attention_lstm_dim = cfg.attention_lstm_dim
        self.encoder_lstm_dim = cfg.encoder_lstm_dim
        self.attention_dropout_p = cfg.attention_dropout_p
        self.decoder_dropout_p = cfg.decoder_dropout_p

        self.max_decoder_steps = cfg.max_decoder_steps
        self.gate_threshold = cfg.gate_threshold
        
        self.prenet = Prenet(
            cfg.n_mel_channels * cfg.n_frames_per_step,
            [cfg.prenet_dim, cfg.prenet_dim],
            dropout_p=cfg.prenet_dropout_p,
            activation=cfg.prenet_activation
        )

        self.attention_rnn = nn.LSTMCell(
            cfg.prenet_dim + cfg.encoder_lstm_dim,
            cfg.attention_lstm_dim)

        self.attention_layer = Attention(
            attention_lstm_dim=cfg.attention_lstm_dim, encoder_lstm_dim=cfg.encoder_lstm_dim,
            attention_dim=cfg.attention_dim, attention_n_filters=cfg.attention_n_filters,
            attention_kernel_size=cfg.attention_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            cfg.attention_lstm_dim + cfg.encoder_lstm_dim,
            cfg.decoder_lstm_dim)

        self.linear_projection = LinearNorm(
            cfg.decoder_lstm_dim + cfg.encoder_lstm_dim,
            cfg.n_mel_channels * cfg.n_frames_per_step)

        self.gate_layer = LinearNorm(
            cfg.decoder_lstm_dim + cfg.encoder_lstm_dim, 1,
            bias=True, w_init_gain='sigmoid')


    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_lstm_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_lstm_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_lstm_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_lstm_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_lstm_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.attention_dropout_p, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.decoder_dropout_p, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, encoder_outputs, decoder_inputs, text_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        text_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        #print("decoder_inputs.shape", decoder_inputs.shape)

        decoder_input = self.get_go_frame(encoder_outputs).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            encoder_outputs, mask=~get_mask_from_lengths(text_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, encoder_outputs):
        """ Decoder inference
        PARAMS
        ------
        encoder_outputs: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(encoder_outputs)

        self.initialize_decoder_states(encoder_outputs, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

