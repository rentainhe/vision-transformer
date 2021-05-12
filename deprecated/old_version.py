from deprecated.build_config import configs
import argparse
import yaml

def parse_option():
    parser = argparse.ArgumentParser('Vision Transformer training and evaluation script', add_help=False)
    parser.add_argument('--CONFIG_FILE', type=str, required=True, metavar="FILE", help='path to config file', )

    # easy config modification
    parser.add_argument('--BATCH_SIZE', type=int, help="batch size for single GPU")
    parser.add_argument('--DATA_PATH', type=str, help='path to dataset')
    parser.add_argument('--RESUME', help='resume from checkpoint')
    parser.add_argument('--ACCUMULATION_STEPS', type=int, help="gradient accumulation steps")
    parser.add_argument('--USE_CHECKPOINT', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--OUTPUT', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--TAG', help='tag of experiment')
    parser.add_argument('--EVAL', action='store_true', help='Perform evaluation only')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_option()
    # Load Base Config
    base_cfg = "configs/base.yaml"
    with open(base_cfg, 'r') as f:
        base_dict = yaml.load(f, Loader=yaml.FullLoader)
    configs.add_args(base_dict)

    # Load Model Specific Config
    model_cfg = "{}".format(args.CONFIG_FILE)
    with open(model_cfg, 'r') as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Load Args Config
    args_dict = configs.parse_to_dict(args)
    args_dict = {**model_dict, **args_dict}
    configs.add_args(args_dict)

    print("Hyper Parameters:")
    print(configs)
    print(configs.MODEL)
    print(tuple(eval(configs.MODEL['T2T_VIT']['T2T_LAYERS'])))

    # test model
    from models.build import build_model
    model = build_model(configs)
    import torch
    test = torch.randn(1,3,224,224)
    print(model(test).size())