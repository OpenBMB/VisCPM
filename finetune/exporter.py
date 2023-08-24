import os
import shutil
import torch
import torch.distributed
from deepspeed.utils import logger

from finetune.utils import utils


def export(vllm_engine, global_step, epoch, args):
    # Save the checkpoint for training recovery
    logger.info(f'start to deepspped ckpt, save_dir={args.exp_ckpt_dir}')
    vllm_engine.save_checkpoint(save_dir=args.exp_ckpt_dir, tag=f'global_step{global_step}', client_state={
        'checkpoint_step': global_step, 'epoch': epoch})

    # Export the model and related data for later use
    export_model_dir = args.exp_model_dir
    os.makedirs(export_model_dir, exist_ok=True)
    base_file_name = f'{args.exp_name}_{global_step}'

    # model files
    if utils.is_main_process():
        model_state_dict_path = os.path.join(export_model_dir, base_file_name + '.pt')
        model_cfg_path = os.path.join(export_model_dir, 'config.json')
        paths = [model_state_dict_path, model_cfg_path]

        torch.save(vllm_engine.module.state_dict(), model_state_dict_path)
        shutil.copy(args.llm_path, model_cfg_path)

        logger.info(f'Successfully save model files!  {paths}')
    torch.distributed.barrier()
