from .vit import ViT
from .t2t import T2T_ViT
from .swin_transformer import SwinTransformer

def build_model(config):
    model_type = config.MODEL['TYPE']
    if model_type == 'vit':
        model = ViT(image_size=config.DATA.IMG_SIZE,
                    patch_size=config.MODEL.VIT.PATCH_SIZE,
                    num_classes=config.MODEL.VIT.NUM_CLASSES,
                    dim=config.MODEL.VIT.EMBED_DIM,
                    depth=config.MODEL.VIT.DEPTH,
                    heads=config.MODEL.VIT.HEADS,
                    dim_head=config.MODEL.VIT.DIM_HEAD,
                    mlp_dim=config.MODEL.VIT.MLP_DIM,
                    dropout=config.MODEL.VIT.DROPOUT,
                    emb_dropout=config.MODEL.VIT.EMB_DROPOUT,
                    pool=config.MODEL.VIT.POOL)

    elif model_type == 't2t-vit':
        model = T2T_ViT(image_size=config.DATA.IMG_SIZE,
                        num_classes=config.MODEL.T2T_VIT.NUM_CLASSES,
                        dim=config.MODEL.T2T_VIT.EMBED_DIM,
                        depth=config.MODEL.T2T_VIT.DEPTH,
                        heads=config.MODEL.T2T_VIT.HEADS,
                        dim_head=config.MODEL.T2T_VIT.DIM_HEAD,
                        mlp_dim=config.MODEL.T2T_VIT.MLP_DIM,
                        dropout=config.MODEL.T2T_VIT.DROPOUT,
                        emb_dropout=config.MODEL.T2T_VIT.EMB_DROPOUT,
                        pool=config.MODEL.T2T_VIT.POOL,
                        tokens_type=config.MODEL.T2T_VIT.TOKENS_TYPE,
                        t2t_layers=config.MODEL.T2T_VIT.T2T_LAYERS)
    elif model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unknow model: {model_type}")

    return model