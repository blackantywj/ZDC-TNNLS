import os
import wandb
import random
import numpy as np
import torch
import ml_collections
import torch.distributed as dist
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
from tqdm.auto import trange
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler

from functools import partial

import json
from copy import deepcopy
import heapq

from diffusion_utils.dynamic import DynamicSDE, DDPMDynamic
from diffusion_utils.solvers import create_solver, EulerSolverX0, DDIMSolver

from utils.ema_model import ExponentialMovingAverage
from utils.util import mse_loss, get_stat, reduce_tensor, set_seed, get_uniform_ball_noise
from data.dataset import DatasetDDP, get_dataset_iter
from data.util import tokenize, BatchEncoding

from model.score_estimator import ScoreEstimatorEMB
from model.encoder import Encoder
from model.enc_normalizer import EncNormalizer
from model.decoder import Decoder

from estimation_utils.util import gather_texts, compute_metric
from estimation_utils.metrics import compute_metric

import clip, math

import pickle

from diffusion_utils.dpm_solver_pytorch import model_wrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, _ = clip.load("ViT-B/32", device=device, jit=False)  # load model & ignore image preprocess
token = clip.tokenize(["", "", "", "", "",]).to(device)
# with torch.no_grad():

class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config

        # Diffusion Encoder
        gen_cfg = config.model.encoder_link
        self.tokenizer = AutoTokenizer.from_pretrained(gen_cfg)
        if "gpt2" in gen_cfg:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        if not config.emb:
            self.gen_enc_normalizer = EncNormalizer(
                enc_mean_path=self.config.data.enc_gen_mean,
                enc_std_path=self.config.data.enc_gen_std,
            )
        else:
            self.gen_enc_normalizer = None
        
        self.encoder = Encoder(
            gen_cfg,
            enc_normalizer=self.gen_enc_normalizer,
            is_change_sp_tokens=True,
            emb=config.emb
        ).eval().cuda()

        # Decoder
        self.decoder = Decoder(
            decoder_config=config.decoder,
            diffusion_config=config.se_config
        )
        self.restore_decoder(config)
        self.decoder = self.decoder.cuda().eval()
        
        # Score estimator
        self.se_config = deepcopy(config.se_config)
        self.se_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(
            config=self.se_config,
            ssf = self.config.model.ssf
        ).cuda()

        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )

        # Number of parameters
        self.config.params_number = ml_collections.ConfigDict()
        self.config.params_number.score_estimator = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.params_number.decoder = sum(p.numel() for p in self.decoder.parameters())
        self.config.params_number.generative_encoder = sum(p.numel() for p in self.encoder.parameters())

        self.device = next(self.score_estimator.parameters()).device

        # Dynamic
        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling
        )   

        # self.dynamic = DDPMDynamic(config)
        # self.diff_eq_solver = EulerSolverX0(
        #     dynamic=self.dynamic,
        #     score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
        #     ode_sampling=config.training.ode_sampling
        # )        

        if not config.eval:
            self.train_datasets_iter = DatasetDDP(
                split="train",
                config=config,
            ).get_data()
            self.train_dataset = None

        # Checkpoint loading
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)
        
        try:
            self.valid_dataset = next(DatasetDDP(split="test", config=config).get_data())
        except Exception:
            self.valid_dataset = next(DatasetDDP(split="test", config=config).get_data())

        # Checkpoint utils
        self.all_checkpoints = []
        self.tracked_test_metric = dict() # step -- metric value
        
        if self.config.ddp and dist.get_rank() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.training.checkpoints_prefix,
                config=dict(self.config),
                mode="offline",
            )
        
        if eval:
            self.restore_parameters(self.device)
            self.score_estimator.eval()
            self.estimate("test")
        else:
            self.set_optimizer()
            self.set_scheduler()
            self.set_grad_scaler()
            self.step = 0
            
            if self.load_checkpoint():
                # if self.config.is_conditional:
                #     self.estimate("test")
                self.estimate("test")
                self.validate()

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        
        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False
            
        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"
        load = torch.load(checkpoint_name)
        self.step = load["step"]
        self.ema.load_state_dict(load["ema"])
        self.switch_to_ema()
        
    def save_checkpoint(self, last: bool = False) -> None:
        if not dist.get_rank() == 0:
            return

        if not os.path.exists(self.config.training.checkpoints_folder):
            os.makedirs(self.config.training.checkpoints_folder)
            
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")

        if self.config.higher_better:
            item = (self.tracked_test_metric[self.step], save_path)
        else:
            item = (-self.tracked_test_metric[self.step], save_path)

        if self.config.save_top_k is None or self.config.save_top_k > len(self.all_checkpoints):
            self.__save_checkpoint(save_path)
            heapq.heappush(self.all_checkpoints, item)
        else:
            heap_smallest = self.all_checkpoints[0]
            if heap_smallest[0] < item[0]:
                self.__remove_checkpoint(heap_smallest[1])
                heapq.heappop(self.all_checkpoints)
                self.__save_checkpoint(item[1])
                heapq.heappush(self.all_checkpoints, item)

    def __remove_checkpoint(self, save_path):
        os.remove(save_path)
    
    def __save_checkpoint(self, save_path):
        torch.save(
            {
                "model": self.score_estimator.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.grad_scaler.state_dict(),
                "step": self.step,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            save_path
        )
        print(f"Save model to: {save_path}")
        
    def load_checkpoint(self) -> int:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False
            
        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.score_estimator.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        
        self.step = load["step"]
        if dist.get_rank() == 0:
            print(f"Checkpoint is loaded {checkpoint_name}")
        return True

    def restore_decoder(self, config):
        decoder_path = self.config.decoder.decoder_path
        if config.decoder.negative_train:
            originname = self.config.decoder.decoder_path.split('/')[-1].split('.')[0]
            decoder_path = f"datasets/{self.config.decoder.decoder_path.split('/')[-2]}/{originname}_ssf_{config.model.ssf}_tmp_{self.config.decoder.temperature}_ADD_HID_{str(self.config.decoder.add_hid)}_ADD_ATT_{str(self.config.decoder.add_att)}_epoch_{self.config.decoder.epochs}_nl_{config.decoder.num_hidden_layers}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_centerpos_{config.decoder.centerpos}_{self.config.decoder.percent}.pth"
            print(f'load the model: {decoder_path}')
        else:
            originname = self.config.decoder.decoder_path.split('/')[-1].split('.')[0]
            decoder_path = f"datasets/{self.config.decoder.decoder_path.split('/')[-2]}/{originname}_ssf_{config.model.ssf}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_nl_{config.decoder.num_hidden_layers}.pth"
        self.decoder.load_state_dict(torch.load(decoder_path)["decoder"])

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def collate_fn(self, batch, split):
        texts_trg = [t["text_trg"] for t in batch]
        tok_trg = self.tokenizer(
            texts_trg,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_sequence_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        
        if self.config.is_conditional:
            if split == 'train':
                texts_src = [t["text_src"] for t in batch]
                tok_src = self.tokenizer(
                    texts_src,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_context_len,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                text_trg_cond = [t["text_trg_cond"] for t in batch]
                # tokenize → Tensor shape (N, 77)
                text_trg_cond_tokens = clip.tokenize(text_trg_cond, truncate=True)
                rt_captions = torch.stack([t['rt_captions'] for t in batch])
                # 可选：L2 归一化

                new_batch = {
                    "text_src": texts_src,
                    "input_ids_src": tok_src["input_ids"],
                    "attention_mask_src": tok_src["attention_mask"],
                    "text_trg": texts_trg,
                    "input_ids_trg": tok_trg["input_ids"],
                    "attention_mask_trg": tok_trg["attention_mask"],
                    "text_trg_cond_tokens": text_trg_cond_tokens,
                    "rt_captions": rt_captions,
                }
                
                if "references" in batch[0]:
                    new_batch["text_references"] = [t["references"] for t in batch]

                new_batch = BatchEncoding(new_batch)
            else:
                texts_src = [t["text_src"] for t in batch]
                tok_src = self.tokenizer(
                    texts_src,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_context_len,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )                
                # tokenize → Tensor shape (N, 77)
                rt_captions = torch.stack([t['rt_captions'] for t in batch])
                # 可选：L2 归一化
                image_features = torch.stack([torch.tensor(i['img_features_list']) for i in batch]).float()
                image_features /= image_features.norm(2, dim = -1, keepdim = True)
                if isinstance(batch[0]['cocoid_list'], str):
                    cocoidlist = torch.tensor([int(i['cocoid_list'].split('.')[0]) for i in batch])
                else:
                    cocoidlist = torch.tensor([int(i['cocoid_list']) for i in batch])
                new_batch = {
                    "text_src": texts_src,
                    "input_ids_src": tok_src["input_ids"],
                    "attention_mask_src": tok_src["attention_mask"],
                    "text_trg": texts_trg,
                    "input_ids_trg": tok_trg["input_ids"],
                    "attention_mask_trg": tok_trg["attention_mask"],
                    "rt_captions": rt_captions,
                    "cocoidlist": cocoidlist,
                    "image_features": image_features,
                }
                if "references" in batch[0]:
                    new_batch["text_references"] = [t["references"] for t in batch]

                new_batch = BatchEncoding(new_batch)
        else:
            new_batch = BatchEncoding({
                "text_trg": texts_trg,
                "input_ids_trg": tok_trg["input_ids"],
                "attention_mask_trg": tok_trg["attention_mask"],
            })
        return new_batch

    def set_train_data_generator(self) -> None:
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)
        print("Dataset length:", len(self.train_dataset))

        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=16,
            batch_size=self.config.training.batch_size_per_gpu,
            shuffle=True,
            collate_fn=partial(self.collate_fn, split = 'train'),
        )

    def set_valid_data_generator(self) -> None:
        self.valid_loader = DataLoader(
            self.valid_dataset,
            num_workers=16,
            batch_size=self.config.validation.batch_size,
            collate_fn=partial(self.collate_fn, split = 'val'),
            shuffle=False,
        )

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        # loss.backward()
        grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad])
        )

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # My custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.FloatTensor(batch_size).uniform_() * (self.dynamic.T - eps) + eps
    
    def train(self) -> None:
        self.set_valid_data_generator()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()
            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, batch in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(batch)

            if self.step % self.config.training.eval_freq == 0:
                if self.config.is_conditional: 
                    self.estimate("test")
                # self.estimate("test")
                self.validate()
            
            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.accum_batch_steps == 0:
                self.train_range.set_description(
                    f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                    f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                )

    def train_step(self, batch):
        self.step += 1

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            batch = batch.to(f"cuda:{dist.get_rank()}")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
                batch['text_features'] = clipmodel.encode_text(batch['text_trg_cond_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                batch['text_features'] = self.noise_injection(batch['text_features'], variance = 0.016, device=device)
            else:
                src_x = None

            trg_x = self.encoder(**{
                "input_ids": batch["input_ids_trg"], 
                "attention_mask": batch["attention_mask_trg"]
            })
        if self.config.model.ssf:
            text_features = self.ddp_score_estimator.module.linear(batch['text_features']).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
            prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)

            rt_captions = self.ddp_score_estimator.module.rt_linear(batch['rt_captions'].to(device))
            att_gt_n_rt = self.ddp_score_estimator.module.crossatt(text_features, rt_captions)

            inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
            loss_dict, stat_dict = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch, cond_embed = cond_embed, cond_embed_mask=cond_embed_mask)

        else:
            text_features = self.ddp_score_estimator.module.linear(batch['text_features']).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
            prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)
            inputs = torch.cat((text_features, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
            loss_dict, stat_dict = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch, cond_embed = cond_embed, cond_embed_mask=cond_embed_mask)
        
        if self.step % self.config.training.accum_batch_steps == 0:
            stat_dict["grad_norm"] = self.optimizer_step(loss_dict['total_loss'])
            stat_dict["scale_factor"] = torch.Tensor([self.grad_scaler._scale])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric("statistics", k, v.item())

        return loss_dict, stat_dict

    @torch.no_grad()
    def validate(self) -> None:
        self.set_valid_data_generator()
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        for batch in self.valid_loader:
            batch = batch.to(f"cuda:{dist.get_rank()}")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
                # batch['text_features'] = clipmodel.encode_text(batch['text_trg_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                # batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                text_features = self.ddp_score_estimator.module.linear(batch['image_features']).view(batch['image_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
                prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)
                if self.config.model.ssf:
                    rt_captions = self.ddp_score_estimator.module.rt_linear(batch['rt_captions'].to(device))
                    att_gt_n_rt = self.ddp_score_estimator.module.crossatt(text_features, rt_captions)

                    inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
                    cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
                    cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
                else:
                    inputs = torch.cat((text_features, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
                    cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
                    cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
            else:
                src_x = None

            trg_x = self.encoder(**{
                "input_ids": batch["input_ids_trg"], 
                "attention_mask": batch["attention_mask_trg"]
            })
            
            loss_dict, _ = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch, cond_embed = cond_embed, cond_embed_mask = cond_embed_mask)
            for k, v in loss_dict.items():
                if k in valid_loss:
                    valid_loss[k] += v.item() * trg_x.size(0)
                else:
                    valid_loss[k] = torch.Tensor([v.item() * trg_x.size(0)])
            valid_count += trg_x.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def predict_x_0_unconditional(
        self,
        model,
        x_t, t,
        attention_mask=None,
        x_0_self_cond=None,
    ) -> torch.Tensor:
        texts_src = ["" for _ in range(x_t.shape[0])]
        tok_src = self.tokenizer(
            texts_src,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_context_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(f"cuda:{dist.get_rank()}")
        src_x = self.encoder(
            input_ids=tok_src["input_ids"],
            attention_mask=tok_src["attention_mask"]
        )
        src_text_tokens = clip.tokenize(texts_src, truncate=True)
        cond_embed = clipmodel.encode_text(src_text_tokens.to("cuda:0")).to(torch.float32)
        cond_embed /= cond_embed.norm(dim=1, keepdim=True)
        text_features = self.ddp_score_estimator.module.linear(cond_embed).view(cond_embed.shape[0], 10, -1)  # (b, clip_project_length, d_model)
        prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)
        
        if self.config.model.ssf:   
            replace = clipmodel.encode_text(token)
            replace = replace.expand(cond_embed.shape[0], -1, -1).to(torch.float32).to(device)
            rt_captions = self.ddp_score_estimator.module.rt_linear(replace)
            att_gt_n_rt = self.ddp_score_estimator.module.crossatt(text_features, rt_captions)

            inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            # cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")                  
        else:
            inputs = torch.cat((text_features, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            cond_embed = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
        
        x_0 = model(
            x_t=x_t, 
            time_t=t, 
            cond=src_x,
            attention_mask=attention_mask, 
            cond_mask=None,
            x_0_self_cond=x_0_self_cond,
            cond_embed = cond_embed,
            cond_embed_mask = None
        )
        return x_0
        
    def calc_score(
            self,
            model,
            x_t, t,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None,
            cond_embed = None, 
            cond_embed_mask = None
    ) -> Dict[str, torch.Tensor]:
        """
        x_0 - prediction x_0(x_t, t)
        eps = (x_t - sqrt(alpha_t) * x_0) / std
        score = (-x_t + sqrt(alpha_t) * x_0) / std**2
        """
        params = self.dynamic.marginal_params(t)
        x_0 = model(
            x_t=x_t, time_t=t, cond=cond,
            attention_mask=attention_mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond,
            cond_embed = cond_embed,
            cond_embed_mask = cond_embed_mask
        )
        
        if not model.training and self.config.validation.cfg_coef and self.config.is_conditional:
            x_0_null = self.predict_x_0_unconditional(model, x_t=x_t, t=t, attention_mask=attention_mask, x_0_self_cond=x_0_self_cond)
            x_0 = x_0 + self.config.validation.cfg_coef * (x_0 - x_0_null)
        
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }

    def calc_loss(
            self,
            clean_x,
            cond_x,
            batch=None,
            cond_embed=None,
            cond_embed_mask=None,
            eps: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mask = None  # X["input_mask"]
        
        # Noizing
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if self.config.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                t_next = t
                params_next = self.dynamic.marginal_params(t_next)
                x_t_next = params_next["mu"] * clean_x + params_next["std"] * noise
                with torch.no_grad():
                    x_0_self_cond = self.ddp_score_estimator(
                        x_t=x_t_next, time_t=t_next, cond=cond_x,
                        attention_mask=mask, 
                        cond_mask=batch.get("attention_mask_src"), 
                        x_0_self_cond=x_0_self_cond, 
                        cond_embed = cond_embed,
                        cond_embed_mask = cond_embed_mask,
                    ).detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_0 = self.ddp_score_estimator(
                x_t=x_t, time_t=t, cond=cond_x,
                attention_mask=mask, 
                cond_mask=batch.get("attention_mask_src"),
                x_0_self_cond=x_0_self_cond,
                cond_embed = cond_embed,
                cond_embed_mask = cond_embed_mask,
            )

        # MSE losses
        loss_x_0 = mse_loss(clean_x, x_0, mask)

        loss_dict = {
            'total_loss': loss_x_0,
            'loss_x_0': loss_x_0,
        }

        with torch.no_grad():
            stat_dict = {}
            clean_x_dict = get_stat(clean_x, mask)
            for key in clean_x_dict:
                stat_dict[f"clean_x_{key}"] = clean_x_dict[key]
    
            x_0_dict = get_stat(x_0.detach(), mask)
            for key in x_0_dict:
                stat_dict[f"x_0_{key}"] = x_0_dict[key]
    
            mask = batch["attention_mask_trg"]
            clean_x_dict_SPT = get_stat(clean_x, mask)
            for key in clean_x_dict_SPT:
                stat_dict[f"clean_x_woSPT_{key}"] = clean_x_dict_SPT[key]
    
            x_0_dict_SPT = get_stat(x_0, mask)
            for key in x_0_dict_SPT:
                stat_dict[f"x_0_woSPT_{key}"] = x_0_dict_SPT[key]

        return loss_dict, stat_dict

    @torch.no_grad()
    def generate_text_conditional(self, dataset_name: str, split: str):
        dt = next(get_dataset_iter(self.config, dataset_name, split=split))
        loader = DataLoader(
            dt,
            num_workers=4,
            batch_size=self.config.validation.batch_size,
            collate_fn=partial(self.collate_fn, split = split),
        )
        
        result_dict = {
            "GEN": [],
            "TRG": []
        }
        if self.config.is_conditional:
            result_dict["SRC"] = []
            result_dict["cocoid"] = []
        for batch in loader:
            if dist.is_initialized():
                batch = batch.to(f"cuda:{dist.get_rank()}")
            else:
                batch = batch.to(f"cuda:0")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
                # batch['text_features'] = clipmodel.encode_text(batch['text_trg_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                # batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                text_features = self.ddp_score_estimator.module.linear(batch['image_features']).view(batch['image_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
                prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)

                rt_captions = self.ddp_score_estimator.module.rt_linear(batch['rt_captions'].to(device))
                att_gt_n_rt = self.ddp_score_estimator.module.crossatt(text_features, rt_captions)

                inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
                fusion_features = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
                # fusion_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
                # batch['text_features'] = clipmodel.encode_text(batch['texts_trg_cond_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                # batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                # # batch["text_features"] = self.noise_injection(batch["text_features"], 0.016)
                # batch["text_features_mask"] = torch.ones(batch["text_features"].shape[0], 10, dtype=torch.int64, device="cuda:0")
                # # src_x = torch.concat((batch["text_features"], src_x), dim = 1)      
            else:
                src_x = None
            
            ###########################
            #    test code
            trg_latent = self.encoder(**{
                "input_ids": batch["input_ids_trg"],
                "attention_mask": batch["attention_mask_trg"]
            })            
            ###########################


            gen_text = self.generate_text_batch(
                batch_size=len(batch["text_trg"]),
                cond_x=src_x,
                cond_mask=None,
                cond_embed = fusion_features,
                cond_embed_mask = None,
                trg_latent = trg_latent,
                batch=batch
            )[0]

            if dataset_name not in self.config.data.datasets.downstream_tasks:
                result_dict["TRG"] += self.tokenizer.batch_decode(batch["input_ids_trg"], skip_special_tokens=True)
            else:
                if "text_references" in batch:
                    result_dict["TRG"] += batch["text_references"]
                else:
                    result_dict["TRG"] += batch["text_trg"]

            result_dict["GEN"] += gen_text
            result_dict["cocoid"] += batch['cocoidlist'].tolist()
            if self.config.is_conditional:
                result_dict["SRC"] += batch["text_src"]

            if len(result_dict["TRG"]) >= (self.config.validation.num_gen_texts // dist.get_world_size()):
                break

        return result_dict

    def generate_text_conditional_5000(self, dataset_name: str, split: str):
        dt_raw = next(get_dataset_iter(self.config, dataset_name, split=split))
        # dt_raw = next(get_dataset_iter(self.config, dataset_name, split=split))
        unique_ds = UniqueCocoIterableDataset(dt_raw, max_cocoids=5000)
        loader = DataLoader(
            unique_ds,
            num_workers=2,
            batch_size=self.config.validation.batch_size,
            collate_fn=partial(self.collate_fn, split = split),
        )

        result_dict = {"GEN": [], "TRG": [], "cocoid": []}
        if self.config.is_conditional:
            result_dict["SRC"] = []

        seen_cocoids = set()  # 用来记录哪些 cocoid 已处理

        for batch in loader:
            batch_cocoids = batch["cocoidlist"].tolist()
            # 找到 batch 中还没处理过的索引
            new_idx = [i for i, cid in enumerate(batch_cocoids) if cid not in seen_cocoids]
            if not new_idx:
                continue  # 全部都是重复的，跳过

            # 更新 seen_cocoids
            for i in new_idx:
                seen_cocoids.add(batch_cocoids[i])

            # 对 batch 的各个字段做子集筛选，只保留 new_idx
            def subset(x):
                # 如果是 tensor，就用 index_select；如果是 list，就用列表推导
                if isinstance(x, torch.Tensor):
                    return x.index_select(0, torch.tensor(new_idx, device=x.device))
                elif not isinstance(x[0], str):
                    return torch.tensor([x[i] for i in new_idx])
                else:
                    return [x[i] for i in new_idx]
            # 重新构建一个小 batch
            small_batch = {k: subset(v) for k, v in batch.items()}

            # 选 device
            if dist.is_initialized():
                device = f"cuda:{dist.get_rank()}"
            else:
                device = "cuda:0"

            # 把 small_batch 中的每个 tensor 都搬到对应的 device
            for k, v in small_batch.items():
                if not isinstance(v, list):
                    small_batch[k] = v.to(device)
                else:
                    small_batch[k] = v

            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": small_batch["input_ids_src"],
                    "attention_mask": small_batch["attention_mask_src"]
                })
                # batch['text_features'] = clipmodel.encode_text(batch['text_trg_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                # batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                text_features = self.ddp_score_estimator.module.linear(small_batch['image_features']).view(small_batch['image_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
                prefix = self.ddp_score_estimator.module.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.ddp_score_estimator.module.prefix_const.shape)  # (b, prefix_length, d_model)
                
                if self.config.model.ssf:
                    rt_captions = self.ddp_score_estimator.module.rt_linear(small_batch['rt_captions'].to(device))
                    att_gt_n_rt = self.ddp_score_estimator.module.crossatt(text_features, rt_captions)

                    inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
                    fusion_features = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
                # fusion_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
                # batch['text_features'] = clipmodel.encode_text(batch['texts_trg_cond_tokens'].to("cuda:0")).to(torch.float32)  # → (N,512)
                # batch['text_features'] /= batch['text_features'].norm(dim=1, keepdim=True)
                # # batch["text_features"] = self.noise_injection(batch["text_features"], 0.016)
                # batch["text_features_mask"] = torch.ones(batch["text_features"].shape[0], 10, dtype=torch.int64, device="cuda:0")
                # # src_x = torch.concat((batch["text_features"], src_x), dim = 1)     
                else:
                    inputs = torch.cat((text_features, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
                    fusion_features = self.ddp_score_estimator.module.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            else:
                src_x = None

            trg_latent = self.encoder(**{
                "input_ids": small_batch["input_ids_trg"],
                "attention_mask": small_batch["attention_mask_trg"]
            })

            gen_text = self.generate_text_batch(
                batch_size=len(small_batch["text_trg"]),
                cond_x=src_x,
                cond_mask=None,
                cond_embed=fusion_features,
                cond_embed_mask=None,
                trg_latent=trg_latent,
                batch=small_batch
            )[0]

            # 按原来逻辑把结果 append
            if dataset_name not in self.config.data.datasets.downstream_tasks:
                result_dict["TRG"] += self.tokenizer.batch_decode(
                    small_batch["input_ids_trg"], skip_special_tokens=True
                )
            else:
                if "text_references" in small_batch:
                    result_dict["TRG"] += small_batch["text_references"]
                else:
                    result_dict["TRG"] += small_batch["text_trg"]

            result_dict["GEN"] += gen_text
            result_dict["cocoid"] += [batch_cocoids[i] for i in new_idx]
            if self.config.is_conditional:
                result_dict["SRC"] += small_batch["text_src"]

            # 可以按需加个终止条件
            if len(result_dict["TRG"]) >= (self.config.validation.num_gen_texts // dist.get_world_size()):
                break

        return result_dict

    @torch.no_grad()
    def generate_text_batch(self, batch_size, cond_x=None, attention_mask=None, cond_mask=None, cond_embed = None, cond_embed_mask = None, trg_latent = None, batch = None):
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        import time
        # s = time.time()
        pred_embeddings = self.pred_embeddings(
            batch_size = batch_size,
            attention_mask = attention_mask,
            cond_x = cond_x,
            cond_mask = cond_mask,
            cond_embed = cond_embed, 
            cond_embed_mask = cond_embed_mask
        )
        if self.config.dynamic.solver == "dpmsolver":
            pred_embeddings = self.pred_embeddings_ds(
                batch_size = batch_size,
                attention_mask = attention_mask,
                cond_x = cond_x,
                cond_mask = cond_mask,
                cond_embed = cond_embed, 
                cond_embed_mask = cond_embed_mask
            )
        # cond_x = torch.concat((cond_embed, cond_x), dim = 1)
        # e1 = time.time()
        # print("e1 - s:", e1 - s)
        output = self.pred_logits(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask, batch=batch)
        tokens = output.argmax(dim=-1)        
        e = time.time()

        # print("e - s:", e - s)
        end_tokens = []
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.eos_token])
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.sep_token])

        tokens = tokens.detach().cpu().tolist()
        tokens_list = []
        for seq in tokens:
            id = 0
            while id < len(seq) and seq[id] not in end_tokens:
                id += 1
            tokens_list.append(seq[0: id])

        text = self.tokenizer.batch_decode(tokens_list, skip_special_tokens=True)

        
        return text, pred_embeddings

    @torch.no_grad()
    def pred_logits(self, pred_embeddings, cond_x=None, cond_mask=None, cond_embed = None, cond_embed_mask = None, batch = None):
        if not self.config.emb:
            pred_embeddings = self.gen_enc_normalizer.denormalize(pred_embeddings)
            if self.config.decoder.is_conditional and cond_x is not None:
                cond_x = self.gen_enc_normalizer.denormalize(cond_x)
        else:
            cond_x = None
            cond_mask = None
        # cond_embed = self.noise_injection(cond_embed, 0.016)
        # cond_embed = self.decoder.linear(batch['text_features']).view(cond_embed.shape[0], 10, -1)
        # cond_embed_mask = torch.ones(cond_embed.shape[0], 10, dtype=torch.int64, device="cuda:0")       
        text_features = self.decoder.linear(batch['image_features']).view(batch['image_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
        prefix = self.decoder.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *self.decoder.prefix_const.shape)  # (b, prefix_length, d_model)

        rt_captions = self.decoder.rt_linear(batch['rt_captions'].to(device))
        att_gt_n_rt = self.decoder.crossatt(text_features, rt_captions)

        inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
        
        cond_embed = self.decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
        # cond_embed_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")         
        cond_x = torch.concat((cond_embed, cond_x), dim = 1)
        # cond_mask = torch.concat((cond_embed_mask, cond_mask), dim = 1)                  
        output = self.decoder(pred_embeddings, cond_x=cond_x, cond_mask=None)
        return output

    @torch.no_grad()
    def pred_embeddings(
            self,
            batch_size,
            cond_x=None,
            cond_mask=None,
            attention_mask=None,
            cond_embed = None, 
            cond_embed_mask = None
    ) -> torch.Tensor:
        self.score_estimator.eval()
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder.encoder.config.hidden_size,
        )

        with torch.no_grad():
            x = self.dynamic.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
            eps_t = 0.01

            if self.config.timesteps == "linear":
                timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)
            elif self.config.timesteps == "quad":
                deg = 2
                timesteps = torch.linspace(1, 0, self.dynamic.N + 1, device=self.device) ** deg * (self.dynamic.T - eps_t) + eps_t

            for idx in tqdm(range(self.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1]

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=x, t=input_t, next_t=next_input_t,
                    cond=cond_x,
                    cond_mask=cond_mask,
                    attention_mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                    cond_embed = cond_embed, 
                    cond_embed_mask = cond_embed_mask
                )       

                x, x_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]
                
            pred_embeddings = x_mean

        return pred_embeddings

    @torch.no_grad()
    def pred_embeddings_ds(
            self,
            batch_size,
            cond_x=None,
            cond_mask=None,
            attention_mask=None,
            cond_embed = None, 
            cond_embed_mask = None
    ) -> torch.Tensor:
        self.score_estimator.eval()
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder.encoder.config.hidden_size,
        )

        with torch.no_grad():
            x = self.dynamic.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)

            cond_dict = {
                "cond_x" : cond_x,
                "cond_mask" : cond_x,
                "attention_mask" : attention_mask,
                "cond_embed" : cond_embed, 
                "cond_embed_mask" : cond_embed_mask,
                "x_0_self_cond" : x_0_self_cond            
            }

            diff_eq_solver = create_solver(self.config)(
                model_wrapper(self.ddp_score_estimator, self.dynamic, model_type="x_start", model_kwargs=cond_dict),
                self.dynamic,
                algorithm_type=self.config.dynamic.solver
            )     

            output = diff_eq_solver.sample(
                x,
                steps=20,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )       

        return output

    @torch.no_grad()
    def estimate(self, split: str):
        self.score_estimator.eval()
        self.ddp_score_estimator.eval()
        self.switch_to_ema()
        
        result_dict = dict()
        # Generation
        # self.config.data.datasets.datasets_list = ["nocaps"]
        for dataset_name in self.config.data.datasets.datasets_list:
            if dist.is_initialized():
                seed = self.config.seed + self.step + dist.get_rank()
            else:
                seed = self.config.seed + self.step
            set_seed(seed)
            result_dict[dataset_name] = self.generate_text_conditional(dataset_name, split=split)

        # Gathering
        if dist.is_initialized():
            for dataset_name in result_dict:
                for key in result_dict[dataset_name]:
                    result_dict[dataset_name][key] = gather_texts(result_dict[dataset_name][key])
                if dataset_name not in self.config.data.datasets.downstream_tasks:
                    for key in result_dict[dataset_name]:
                        result_dict[dataset_name][key] = result_dict[dataset_name][key][:self.config.validation.num_gen_texts]

        # Logging
        result_list = dict()

        for dataset_name in result_dict:
            keys = list(result_dict[dataset_name].keys())
            result_list[dataset_name] = []
            for ind in range(len(result_dict[dataset_name][keys[0]])):
                result_list[dataset_name].append(
                    {key: result_dict[dataset_name][key][ind] for key in keys}
                )
                
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(self.config.validation.texts_path):
                os.makedirs(self.config.validation.texts_path)

            prefix_folder = os.path.join(self.config.validation.texts_path, self.config.training.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)

            file_name = f"{self.step}-N={self.config.dynamic.N}-len={len(result_list)}_cfg_{self.config.validation.cfg_coef}_ssf_{self.config.model.ssf}_tmp_{self.config.decoder.temperature}_ADD_HID_{str(self.config.decoder.add_hid)}_ADD_ATT_{str(self.config.decoder.add_att)}_epoch_{self.config.decoder.epochs}_nl_{self.config.decoder.num_hidden_layers}_bs_{self.config.decoder.batch_size}_lr_{self.config.decoder.lr}_centerpos_{self.config.decoder.centerpos}_{self.config.decoder.percent}.json"
            save_path = os.path.join(prefix_folder, file_name)
            json.dump(result_list, open(save_path, "w"), indent=4)
            print(f"Texts are saved in {save_path}")

        # Metrics
        metrics_dict = dict()
        for dataset_name in self.config.data.datasets.datasets_list:
            texts_src = result_dict[dataset_name].get("SRC")
            texts_trg = result_dict[dataset_name]["TRG"]
            texts_gen = result_dict[dataset_name]["GEN"]

            metrics_dict[dataset_name] = dict()

            for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                metrics_dict[dataset_name][metric_name] = compute_metric(
                    metric_name, 
                    predictions=texts_gen, 
                    references=texts_trg, 
                    sources=texts_src
                )

        ## Metrics Logging
        if not dist.is_initialized() or dist.get_rank() == 0:
            for dataset_name in self.config.data.datasets.datasets_list:
                print("-----", f"{dataset_name}-{split}", "-----")
                for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                    value = metrics_dict[dataset_name][metric_name]
                    if isinstance(value, dict):
                        for key in value:
                            print(f"{key}: {value[key]:0.5f}")
                            self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=key, value=value[key])
                    else:
                        print(f"{metric_name}: {value:0.5f}")
                        self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=metric_name, value=value)

        if split == "test":
            self.tracked_test_metric[self.step] = metrics_dict[self.config.tracked_dataset][self.config.tracked_metric]
        
        if dist.is_initialized():
            seed = self.config.seed + self.step + dist.get_rank()
        else:
            seed = self.config.seed + self.step
        set_seed(seed)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train()
        self.score_estimator.train()
        
    def noise_injection(self, x, variance = 0.001, device = 'cuda') -> torch.Tensor:
        """
        Args:
            x: tensor with a shape of (batch_size, clip_hidden_size), prefix
            variance: the variance of noise
        Return:
            prefix with noise
        """
        if variance == 0.0:
            return x
        std = math.sqrt(variance)
        # normalization
        x = torch.nn.functional.normalize(x, dim = -1)
        # adding noise
        x = x + (torch.randn(x.shape, device = device) * std)

        return torch.nn.functional.normalize(x, dim = -1)
    
from torch.utils.data import IterableDataset, DataLoader

class UniqueCocoIterableDataset(IterableDataset):
    def __init__(self, base_dataset, max_cocoids):
        self.base = base_dataset
        self.max_cocoids = max_cocoids

    def __iter__(self):
        seen = set()
        for sample in self.base:
            cid = sample["cocoid_list"]
            if cid in seen:
                continue
            seen.add(cid)
            yield sample
            if len(seen) >= self.max_cocoids:
                break