from .vit import ViT
from .t2t import T2T_ViT

def build_model(config):
    model_type = config.MODEL['TYPE']
    if model_type == 'vit':
        model = ViT(image_size=config.IMG_SIZE,
                    patch_size=config.MODEL['VIT']['PATCH_SIZE'],
                    num_classes=config.MODEL['VIT']['NUM_CLASSES'],
                    dim=config.MODEL['VIT']['DIM'],
                    depth=config.MODEL['VIT']['DEPTH'],
                    heads=config.MODEL['VIT']['HEADS'],
                    dim_head=config.MODEL['VIT']['DIM_HEAD'],
                    mlp_dim=config.MODEL['VIT']['MLP_DIM'],
                    dropout=config.MODEL['VIT']['DROPOUT'],
                    emb_dropout=config.MODEL['VIT']['EMB_DROPOUT'],
                    pool=config.MODEL['VIT']['POOL'])
    elif model_type == 't2t-vit':
        model = T2T_ViT(image_size=config.IMG_SIZE,
                        num_classes=config.MODEL['T2T_VIT']['NUM_CLASSES'],
                        dim=config.MODEL['T2T_VIT']['DIM'],
                        depth=config.MODEL['T2T_VIT']['DEPTH'],
                        heads=config.MODEL['T2T_VIT']['HEADS'],
                        dim_head=config.MODEL['T2T_VIT']['DIM_HEAD'],
                        mlp_dim=config.MODEL['T2T_VIT']['MLP_DIM'],
                        dropout=config.MODEL['T2T_VIT']['DROPOUT'],
                        emb_dropout=config.MODEL['T2T_VIT']['EMB_DROPOUT'],
                        pool=config.MODEL['T2T_VIT']['POOL'],
                        tokens_type=config.MODEL['T2T_VIT']['TOKENS_TYPE'],
                        t2t_layers=tuple(eval(config.MODEL['T2T_VIT']['T2T_LAYERS'])))
    else:
        raise NotImplementedError(f"Unknow model: {model_type}")

    return model