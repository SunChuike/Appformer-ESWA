import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.decoder1 import Decoder1, DecoderLayer1

class Informer(nn.Module):
    def __init__(self, app_dim, enc_in, dec_in, app_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()

        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.device = device
        self.app_embed = nn.Embedding(app_out, 64)
        self.dec1_embed = nn.Linear(17, 64, bias=True)
        self.dec2_embed = nn.Linear(81, 64, bias=True)
        self.dec3_embed = nn.Linear(80, 64, bias=True)
        self.dec4_embed = nn.Linear(60, 64, bias=True)

        self.app_embed1 = nn.Embedding(1000, 16)

        self.embedding_params = [(12, 12), (31, 12), (7, 12), (24, 12), (60, 12)]

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim) for num_emb, emb_dim in self.embedding_params])

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)  # 2
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # batch_location_vectors
        self.decoder1 = Decoder1(
            [
                DecoderLayer1(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   64, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   64, n_heads, mix=False),
                    64,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )

        self.projection_app = nn.Linear(d_model, app_out, bias=True)

    def forward(self, app_seq_pred, time_seq, location_vectors, user, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # --------------------数据预处理-------------------------------------------------------
        # 生成enc的输入app序列（预编码+自编码）
        # 得到自编码向量
        app_seq = app_seq_pred[:, :self.label_len, :]

        batch_size, num_time_steps, num_indices = time_seq.shape
        time_seq1 = time_seq.view(-1, num_indices)
        concatenated_tensors = []
        for row in time_seq1:
            embedded_row = [embedding_layer(row_tensor_elem) for row_tensor_elem, embedding_layer in
                            zip(row, self.embedding_layers)]

            concatenated_row = torch.cat([emb.view(1, -1) for emb in embedded_row], dim=1)  # 将每个嵌入向量拼接成一行
            concatenated_tensors.append(concatenated_row)
        # 将拼接后的嵌入向量
        concatenated_tensor1 = torch.cat(concatenated_tensors, dim=0)
        concatenated_tensor = concatenated_tensor1.view(batch_size, num_time_steps, -1)

        T = self.dec4_embed(concatenated_tensor)

        L = location_vectors[:, :self.label_len, :]
        Y = self.app_embed(app_seq.squeeze(-1))  # size:[B, 24, 32]

        user_seq = self.app_embed1(user.squeeze(-1))
        X = torch.cat([Y, user_seq], dim=2)
        X = self.dec3_embed(X)

        L = self.dec1_embed(L)
        Z1 = self.decoder1(X, L)
        Z1 = self.decoder1(Z1, T)

        pred = torch.zeros([Z1.shape[0], self.pred_len, Z1.shape[-1]]).to(self.device)

        Z2 = self.decoder1(Y, L)
        Z2 = torch.cat([Z2, pred], dim=1).float()

        Z1 = self.enc_embedding(Z1)
        enc_out, attns = self.encoder(Z1, attn_mask=enc_self_mask)

        Z2 = self.dec_embedding(Z2)
        dec_out = self.decoder(Z2, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = dec_out[:, -self.pred_len:, :]

        app_out = self.projection_app(dec_out)
        app_out = torch.squeeze(app_out, dim=1)

        if self.output_attention:
            return app_out, attns
        else:
            return app_out
