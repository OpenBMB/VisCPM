import datetime
import gc
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
import time
import deepspeed
import torch
import torch.distributed
import torch.utils.data

from deepspeed.utils import logger
from torch.nn import CrossEntropyLoss

from finetune import initializer
from finetune import exporter
from finetune.utils import utils
from finetune.utils.logger import init_logger
from finetune.utils.utils import interpolate_embed_positions

from VisCPM.models import VLU_CPMBee

logger = init_logger(__name__)
logger.setLevel('INFO')
# logger.setLevel('WARNING')

def get_dataloader(tokenizer, args):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data import DistributedSampler

    from VisCPM.utils.utils import build_transform
    from VisCPM.utils.utils import CPMBeeCollater
    from finetune.dataset.itembuilder import CPMBeeImageTextBuilder
    from finetune.dataset.transformeddataset import TransformedDataset

    try:
        hf_dataset = load_dataset("openbmb/llava_zh")
        data_format = 'hf'
        data_path = hf_dataset
    except Exception as e1:
        logger.warning(f"Failed to load from HuggingFace datasets due to: {e1}")
        try:
            hf_dataset = load_dataset('json', data_files=args.text_path)
            data_format = 'hf'
            data_path = hf_dataset
        except Exception as e2:
            logger.warning(f"Failed to load from local file due to: {e2}")
            raise RuntimeError("Both data loading methods failed!")

    transform = build_transform(is_train=True, input_size=args.img_size)
    builder = CPMBeeImageTextBuilder(
        tokenizer=tokenizer,
        max_len=args.max_len,
        transform=transform,
        query_len=args.query_num,
        extra_inp_dict=None,
        min_resolution=128,
        skip_overlength=True
    )
   
    final_dataset = TransformedDataset(dataset=data_path, builder=builder, local_image_dir=args.image_path, data_format=data_format)
    sampler = DistributedSampler(final_dataset)

    dataloader = DataLoader(
        final_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=CPMBeeCollater(tokenizer=tokenizer, max_len=args.max_len)
    )

    return dataloader


def train(vllm_model, args):
    vllm_model.train()
    vllm_model.vpm.beit3.vision_embed.requires_grad_(False)
    # sft -> tune all
    if not args.tune_vision:
        vllm_model.vpm.beit3.apply(utils.stop_gradient_by_name('A'))

    if not args.tune_llm:
        vllm_model.llm.requires_grad_(False)

    vllm_engine, vllm_optim, _, _ = deepspeed.initialize(
        args=args, model=vllm_model, model_parameters=vllm_model.parameters()
    )

    torch.cuda.synchronize()

    logger.info(f'rank={utils.get_rank()} load model successful')

    tokenizer = initializer.load_llm_tokenizer(args)
    dataloader_train = get_dataloader(tokenizer, args)
    logger.info(f'rank={utils.get_rank()} load dataloader successful')

    global_step = 0
    log_loss = 0
    if args.need_resume:
        load_path, client_state = vllm_engine.load_checkpoint(
            args.exp_ckpt_dir, tag=args.need_resume_tag)
        logger.info(f'Load pre-trained checkpoint from {load_path}, states: {client_state}')
        global_step = client_state['checkpoint_step']
        args.start_epoch = client_state.get('epoch', args.start_epoch)
        logger.info(f'rank={utils.get_rank()} load grad successful')

    # init tensorboard writer
    if args.tensorboard is not None and utils.is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard)
    else:
        writer = None

    loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=-100)
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f'start epoch={epoch}')
        time_monitor = {}
        utils.collect_statsd_metric("init", time_monitor)
        for step, batch in enumerate(dataloader_train):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = batch[k].cuda()
            utils.collect_statsd_metric('dataload', time_monitor)
            vllm_model.zero_grad()
            output = vllm_model(data=batch)

            logits = output.logits.view(-1, output.logits.shape[-1]).contiguous()
            target = batch['target'].view(-1).type(torch.long).contiguous()
            loss = loss_fct(logits, target)
            utils.collect_statsd_metric("forward", time_monitor)

            vllm_engine.backward(loss)
            utils.collect_statsd_metric("backward", time_monitor)

            vllm_engine.step()
            utils.collect_statsd_metric("optim", time_monitor)

            cost_info = f'dataload cost: {(time_monitor["dataload"] - time_monitor["init"]): .2f} ' \
                + f'forward cost {(time_monitor["forward"] - time_monitor["dataload"]): .2f} ' \
                + f'backward cost {(time_monitor["backward"] - time_monitor["forward"]): .2f} ' \
                + f'optim cost {(time_monitor["optim"] - time_monitor["backward"]): .2f}'

            log_loss += loss.item()
            global_step += 1



            if args.tensorboard is not None and utils.is_main_process():
                writer.add_scalar("Loss/train", loss.item(), global_step)

            if global_step % args.log_step == 0:
                log_loss = utils.mean(utils.all_gather(log_loss))
                if utils.is_main_process():
                    logger.info(
                        f'Datetime: {datetime.datetime.now()} Step: { global_step - args.log_step: 6d}-{global_step: 6d}: loss: {log_loss/args.log_step: .4f}')
                    logger.info(f'time cost info {cost_info}')
                log_loss = 0

            if global_step % args.save_step == 0:
                exporter.export(vllm_engine, global_step, epoch, args)

            # end step
            utils.collect_statsd_metric('init', time_monitor)

    # final model
    exporter.export(vllm_engine, global_step, args.epochs-1, args)


def setup_model(args):
    start = time.time()

    llm = initializer.load_llm(args)
    vpm = initializer.load_vpm(args)
    vision_dim = vpm.args.encoder_embed_dim
    model = VLU_CPMBee(llm, vpm, vision_dim, args.query_num)
    if args.model_checkpoint:
        logger.info(f'load model_checkpoint from {args.model_checkpoint}')
        state_dict = torch.load(args.model_checkpoint, map_location='cpu')
        state_dict = interpolate_embed_positions(
            model.vpm, state_dict, pos_embed_key='vpm.beit3.encoder.embed_positions.A.weight')
        model.load_state_dict(state_dict)

        del state_dict
        gc.collect()

    model.cuda()
    torch.cuda.empty_cache()
    end = time.time()
    logger.info(f'rank={utils.get_rank()} load model successful, cost {end-start:.2f}s')
    return model


def main():
    args = initializer.get_args()
    # setup file and device
    initializer.setup(args)
    # load model
    model = setup_model(args)
    # train
    train(model, args)


if __name__ == '__main__':
    main()
