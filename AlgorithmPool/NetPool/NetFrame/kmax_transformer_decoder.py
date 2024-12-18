from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()


def get_norm(name, channels, conv_type='3d'):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name.lower() == 'bn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)
    if name.lower() == 'in':
        return nn.InstanceNorm3d(channels, eps=1e-5, affine=True)

    # if name.lower() == 'bn':
    #     if conv_type == '3d':
    #         return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)
    #     if conv_type == '2d':
    #         return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
    #     if conv_type == '1d':
    #         return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm=None, act=None,
                 conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels, conv_type)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            torch.nn.init.trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            torch.nn.init.trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
        (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)


class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        self._batch_norm_similarity = get_norm('bn', num_heads, conv_type='2d')
        self._batch_norm_retrieved_value = get_norm('bn', channels_v, conv_type='1d')

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value


class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133 + 1):
        super().__init__()
        c_num = 32
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5,
                                                   groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, c_num, kernel_size=1, bias=False, norm='bn',
                                                   act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(c_num, c_num // 2, kernel_size=1, bias=True, norm='bn', act=None)
        torch.nn.init.trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(c_num, c_num // 2, kernel_size=1, bias=False, norm='bn', act=None,
                                             conv_type='1d')
        self._transformer_class_head = ConvBN(c_num, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        torch.nn.init.trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('bn', channels=32)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchwd,bcn->bnhwd',
                                   pixel_space_normalized_feature, cluster_mask_kernel)

        # mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits)

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature}


class kMaXTransformerLayer(nn.Module):
    def __init__(
            self,
            num_classes=133,
            in_channel_pixel=2048,
            in_channel_query=256,
            base_filters=128,
            num_heads=8,
            bottleneck_expansion=2,
            key_expansion=1,
            value_expansion=2,
            drop_path_prob=0.0,
    ):
        super().__init__()

        mid_c_num = 32 * 8

        self._num_classes = num_classes
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='bn', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='bn', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth,
                                         kernel_size=1, bias=False,
                                         norm='bn', act=None, conv_type='1d')
        torch.nn.init.trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                       norm='bn', act=None)
        torch.nn.init.trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                      norm='bn', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, mid_c_num, kernel_size=1, bias=False,
                                              norm='bn', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(mid_c_num, in_channel_query, kernel_size=1, bias=False,
                                          norm='bn', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
                                        in_channel_query=self._bottleneck_channels, num_classes=num_classes)
        self._kmeans_query_batch_norm_retrieved_value = get_norm('bn', self._total_value_depth, conv_type='1d')
        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                             norm='bn', act=None, conv_type='1d', norm_init=0.0)

    def forward(self, pixel_feature, query_feature):
        N, C, H, W, aD = pixel_feature.shape
        _, D, L = query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature))  # N C H W
        query_space = self._query_conv1_bn_act(query_feature)  # N x C x L

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space)  # N C H W
        pixel_value = pixel_value.reshape(N, self._total_value_depth, H * W * aD)
        # k-means assignment.
        prediction_result = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)

        with torch.no_grad():
            clustering_result = prediction_result['mask_logits'].flatten(2).detach()  # N L HW
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result,
                                                 memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        with autocast(enabled=False):
            # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float())  # N x C x L

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
                                                [self._total_key_depth, self._total_key_depth, self._total_value_depth],
                                                dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result


class kMaXTransformerDecoder(nn.Module):
    def __init__(
            self,
            dec_layers: List[int],
            in_channels: List[int],
            num_classes: int,
            num_queries: int,
            drop_path_prob: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
        """
        super().__init__()

        # define Transformer decoder here
        self._kmax_transformer_layers = nn.ModuleList()
        self._num_blocks = dec_layers
        c_num = 32
        os2channels = {32: in_channels[0], 16: in_channels[1], 8: in_channels[2], 4: in_channels[3]}

        for index, output_stride in enumerate([32, 16, 8, 4]):
            for _ in range(self._num_blocks[index]):
                self._kmax_transformer_layers.append(
                    kMaXTransformerLayer(num_classes=num_classes + 1,
                                         in_channel_pixel=os2channels[output_stride],
                                         in_channel_query=c_num,
                                         base_filters=c_num,
                                         num_heads=8,
                                         bottleneck_expansion=1,
                                         key_expansion=1,
                                         value_expansion=1,
                                         drop_path_prob=drop_path_prob)
                )

        self._num_queries = num_queries
        # learnable query features
        self._cluster_centers = nn.Embedding(c_num, num_queries)
        torch.nn.init.trunc_normal_(self._cluster_centers.weight, std=1.0)

        self._class_embedding_projection = ConvBN(c_num, c_num, kernel_size=1, bias=False, norm='bn', act='gelu',
                                                  conv_type='1d')

        self._mask_embedding_projection = ConvBN(c_num, c_num, kernel_size=1, bias=False, norm='bn', act='gelu',
                                                 conv_type='1d')

        self._predcitor = kMaXPredictor(in_channel_pixel=in_channels[-1],
                                        in_channel_query=c_num, num_classes=num_classes + 1)

    def forward(self, x, panoptic_features):
        B = x[0].shape[0]
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1)  # B x C x L

        current_transformer_idx = 0

        # predictions_class = []
        # predictions_mask = []
        # predictions_pixel_feature = []

        for i, feat in enumerate(x):
            for _ in range(self._num_blocks[i]):
                cluster_centers, prediction_result = self._kmax_transformer_layers[current_transformer_idx](
                    pixel_feature=feat, query_feature=cluster_centers
                )
                # predictions_class.append(prediction_result['class_logits'])
                # predictions_mask.append(prediction_result['mask_logits'])
                # predictions_pixel_feature.append(prediction_result['pixel_feature'])
                current_transformer_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)

        # Final predictions.
        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=panoptic_features,
        )
        # predictions_class.append(prediction_result['class_logits'])
        # predictions_mask.append(prediction_result['mask_logits'])
        # predictions_pixel_feature.append(prediction_result['pixel_feature'])

        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1],
        #     'pixel_feature': predictions_pixel_feature[-1],
        # }

        return {
            'pred_logits': prediction_result['class_logits'],
            'pred_masks': prediction_result['mask_logits'],
            'pixel_feature': prediction_result['pixel_feature'],
        }


class kMaXPredictor_new(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133 + 1):
        super().__init__()
        c_num = 32
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5,
                                                   groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, c_num, kernel_size=1, bias=False, norm='bn',
                                                   act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(c_num, c_num // 2, kernel_size=1, bias=True, norm='bn', act=None)
        torch.nn.init.trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(c_num, c_num // 2, kernel_size=1, bias=False, norm='bn', act=None,
                                             conv_type='1d')
        self._transformer_class_head = ConvBN(c_num, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        torch.nn.init.trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('bn', channels=32)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchwd,bcn->bnhwd',
                                   pixel_space_normalized_feature, cluster_mask_kernel)

        # mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits)

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature}


class kMaXTransformerDecoder_new(nn.Module):
    def __init__(
            self,
            dec_layers: List[int],
            in_channels: List[int],
            num_classes: int,
            num_queries: int,
            drop_path_prob: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
        """
        super().__init__()

        # define Transformer decoder here
        self._kmax_transformer_layers = nn.ModuleList()
        self._num_blocks = dec_layers
        c_num = 32
        os2channels = {32: in_channels[0], 16: in_channels[1], 8: in_channels[2]}

        for index, output_stride in enumerate([32, 16, 8]):
            for _ in range(self._num_blocks[index]):
                self._kmax_transformer_layers.append(
                    kMaXTransformerLayer(num_classes=num_classes + 1,
                                         in_channel_pixel=os2channels[output_stride],
                                         in_channel_query=c_num,
                                         base_filters=c_num,
                                         num_heads=8,
                                         bottleneck_expansion=1,
                                         key_expansion=1,
                                         value_expansion=1,
                                         drop_path_prob=drop_path_prob)
                )

        self._num_queries = num_queries
        # learnable query features
        self._cluster_centers = nn.Embedding(c_num, num_queries)
        torch.nn.init.trunc_normal_(self._cluster_centers.weight, std=1.0)

        self._class_embedding_projection = ConvBN(c_num, c_num, kernel_size=1, bias=False, norm='bn', act='gelu',
                                                  conv_type='1d')

        self._mask_embedding_projection = ConvBN(c_num, c_num, kernel_size=1, bias=False, norm='bn', act='gelu',
                                                 conv_type='1d')

        self._predcitor = kMaXPredictor_new(in_channel_pixel=in_channels[-1],
                                        in_channel_query=c_num, num_classes=num_classes + 1)

    def forward(self, x, panoptic_features):
        B = x[0].shape[0]
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1)  # B x C x L

        current_transformer_idx = 0

        # predictions_class = []
        # predictions_mask = []
        # predictions_pixel_feature = []

        for i, feat in enumerate(x):
            for _ in range(self._num_blocks[i]):
                cluster_centers, prediction_result = self._kmax_transformer_layers[current_transformer_idx](
                    pixel_feature=feat, query_feature=cluster_centers
                )
                # predictions_class.append(prediction_result['class_logits'])
                # predictions_mask.append(prediction_result['mask_logits'])
                # predictions_pixel_feature.append(prediction_result['pixel_feature'])
                current_transformer_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)

        # Final predictions.
        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=panoptic_features,
        )
        # predictions_class.append(prediction_result['class_logits'])
        # predictions_mask.append(prediction_result['mask_logits'])
        # predictions_pixel_feature.append(prediction_result['pixel_feature'])

        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1],
        #     'pixel_feature': predictions_pixel_feature[-1],
        # }

        return {
            'pred_logits': prediction_result['class_logits'],
            'pred_masks': prediction_result['mask_logits'],
            'pixel_feature': prediction_result['pixel_feature'],
        }
