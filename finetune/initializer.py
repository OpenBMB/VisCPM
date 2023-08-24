# coding=utf-8

import os
import gc
import time
import glob
import torch
import argparse
import torch.distributed
import numpy as np
from datetime import datetime
from timm import create_model

from cpm_live.tokenizers import CPMBeeTokenizer

from finetune.utils import utils
from finetune.utils.logger import init_logger
from finetune.utils.utils import interpolate_embed_positions
from VisCPM.models import CPMBeeConfig, CPMBeeTorch


logger = init_logger(__name__)


def get_args():  
    parser = argparse.ArgumentParser(
        'VLLM pre-training script', add_help=False)

    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--text_path', default=None, type=str)
    parser.add_argument('--image_path', default=None, type=str)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--save_step', default=100, type=int)
    parser.add_argument('--sft', action='store_true', help='is traing all parameter')
    parser.add_argument('--tune_vision', action='store_true', help='is traing beit3 vision parameter')
    parser.add_argument('--tune_llm', action='store_true', help='is traing llm parameter')

    # Model parameters
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--model_checkpoint', default=None, help='Path to VLLM model to use', type=str)
    parser.add_argument('--model_dir', default=None, help='Model path', type=str)
    parser.add_argument('--exp_name', default=None, help='Model name', type=str)
    parser.add_argument('--data_state_dict_path', default=None, help='Path to dataset state dict', type=str)

    parser.add_argument('--llm_path', default=None, help='Path to LLM model to use', type=str)
    parser.add_argument('--llm_checkpoint', default=None, help='Path to LLM model to use', type=str)
    parser.add_argument('--vpm_path', help='Path to VPM model to use', type=str)
    parser.add_argument('--vpm_checkpoint', help='Path to VPM model to use', type=str)

    
    # deepspeed
    parser.add_argument('--deepspeed_config', default=None, help='Path to deepspeed config to use', type=str)

    # ----- Training -----
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--query_num', default=32, type=int,
                        help='query numbers')
    parser.add_argument('--max_len', default=96, type=int,
                        help='max len')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=5, type=int)


    # 项目训练中断后，标识是否基于之前训练中断时已保存的 deepspeed 参数（梯度、优化器等）续训
    parser.add_argument('--need_resume', action='store_true', default=False,
                        help="resume with deepspeed states")
    parser.add_argument('--need_resume_tag')

    # -----  distributed training parameters -----
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    

    args = parser.parse_args()
    root_dir = './export'  # 
    if not args.exp_name:
        args.exp_name = 'viscpm_sft' 

    # path to save checkpoints. 
    args.exp_ckpt_dir = os.path.join(root_dir, 'checkpoints')
    # final model export path
    args.exp_model_dir = os.path.join(root_dir, 'models')
    args.tensorboard = '{base}/{timestamp}-{export_model_name}'.format(
        base=os.path.join(root_dir, 'logs'),
        timestamp=datetime.now().strftime("%Y%m%d%H%M%S"), export_model_name=args.exp_name)
    
    # 2、文件读取/载入相关
    if args.model_dir:
        model_dir = args.model_dir  
        ckpt_path = _extract_ckpt_path(model_dir)
        if args.sft: 
            args.model_checkpoint = ckpt_path

    if not args.text_path:
        args.text_path = os.path.join(args.data_path, 'llava_instruct_150k_zh.json')
    if not args.image_path:
        args.image_path = os.path.join(args.data_path, 'coco')


    # ----- need_resume -----
    if args.need_resume:
        if not args.need_resume_tag:
            args.need_resume_tag = open(os.path.join(args.exp_ckpt_dir, 'latest'), 'r').read()

    logger.info("args: {args}")
    return args


def _extract_ckpt_path(base_dir: str):
    if base_dir.endswith('.pt'):
            return base_dir
    paths = glob.glob(base_dir + '/*.bin')
    if len(paths) == 0:
        paths = glob.glob(base_dir + '/*.pt')
    elif len(paths) > 0:
        return paths[0]
    else:
        logger.warning(f'WARNING: .pt file not found in base_dir({base_dir})')
        return None

def setup(args):

    # init dist
    utils.init_distributed_mode(args)
    rank = utils.get_rank()

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # init dirs
    necessary_dirs = [args.exp_ckpt_dir, args.exp_model_dir, args.tensorboard]
    if utils.is_main_process():
        for necessary_dir in necessary_dirs:
            if not necessary_dir:
                continue
            os.makedirs(necessary_dir, exist_ok=True)
    logger.info(f"INFO: rank={rank} setup(dirs) done")



# ----------- common loader -----------
def load_llm(args):
    config = CPMBeeConfig.from_json_file(args.llm_path)
    cpm_model = CPMBeeTorch(config)

    if args.llm_checkpoint and not args.model_checkpoint:
        state_dict = torch.load(args.llm_checkpoint)
        cpm_model.load_state_dict(state_dict)
        del state_dict
        gc.collect()

    return cpm_model


def load_llm_tokenizer(args):
    return CPMBeeTokenizer()

def load_vpm(args):
    model = create_model('beit3_large_patch16_224', img_size=args.img_size)

    if args.vpm_checkpoint and not args.model_checkpoint:
        state_dict = torch.load(args.vpm_checkpoint)['model']
        if args.img_size != 224:
            state_dict = interpolate_embed_positions(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        
    return model
