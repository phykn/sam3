import pytest
import torch

from src.ml.components.nn.attention import MultiheadAttention
from src.ml.components.transformer.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
)


@pytest.mark.parametrize("with_padding", [False, True])
def test_spatial_features_match_sequence_features(with_padding):
    layer = TransformerEncoderLayer(
        activation="relu",
        cross_attention=MultiheadAttention(8, 2, batch_first=True),
        d_model=8,
        dim_feedforward=16,
        dropout=0.0,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(8, 2, batch_first=True),
    )
    encoder = TransformerEncoderFusion(
        layer,
        num_layers=1,
        d_model=8,
        num_feature_levels=1,
        add_pooled_text_to_img_feat=False,
    ).eval()
    features = torch.randn(2, 8, 2, 3)
    position = torch.randn_like(features)
    prompt = torch.randn(4, 2, 8)
    padding = torch.zeros(2, 2, 3, dtype=torch.bool)
    padding[:, :, -1] = True
    expected = encoder(
        src=[features.flatten(2).permute(2, 0, 1)],
        src_pos=[position.flatten(2).permute(2, 0, 1)],
        src_key_padding_mask=[padding.flatten(1).T] if with_padding else None,
        prompt=prompt,
        feat_sizes=[(2, 3)],
    )
    actual = encoder(
        src=[features],
        src_pos=[position],
        src_key_padding_mask=[padding] if with_padding else None,
        prompt=prompt,
    )

    torch.testing.assert_close(actual, expected)
