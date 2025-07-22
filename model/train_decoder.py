import os
import torch
import wandb
import numpy as np
import argparse
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.util import dict_to_cuda
from model.decoder import Decoder
from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse

import clip, math

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device, jit=False)  # load model & ignore image preprocess

def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def get_datasets(config):
    train_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="train")
    test_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="test")
    return train_dataset, test_dataset
    

def get_loaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=28,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        next(valid_dataset),
        batch_size=batch_size,
        num_workers=28,
        pin_memory=False,
    )

    return train_loader, valid_loader

def get_train_loaders(train_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=28,
        pin_memory=False,
    )

    return train_loader

def loss_step(batch, tokenizer, encoder, decoder, config, eval=False):
    trg = tokenizer(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=config.decoder.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        ).to("cuda:0")
    targets = trg["input_ids"]
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent = encoder(
            input_ids=targets,
            attention_mask=trg["attention_mask"]
        )
    
    if not eval:
        dynamic = DynamicSDE(config=config)

        if config.decoder.diffusion_forward:
            t = torch.cuda.FloatTensor(latent.shape[0]).uniform_() * (config.decoder.T - config.decoder.eps) + config.decoder.eps
            latent = dynamic.marginal(latent, t)["x_t"]
        else:   
            eps = torch.randn_like(latent) * config.decoder.noise_sigma
            latent = latent + eps
    with torch.no_grad():
        if not config.emb:
            latent = encoder.module.enc_normalizer.denormalize(latent)

    if config.decoder.is_conditional:
        src = tokenizer(
            batch['text_src'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=15,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        ).to("cuda:0")
        # texts_src = [t["text_src"] for t in batch]
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_tokens = clip.tokenize(batch['text_trg'], truncate=True)
            batch['text_features'] = model.encode_text(text_tokens.to("cuda:0"))  # → (N,512)
            batch['text_features'] /= batch['text_features'].norm(2, dim = -1, keepdim = True)
            src_latent = encoder(
                input_ids=src["input_ids"],
                attention_mask=src["attention_mask"]
            )
            if not config.emb:
                src_latent = encoder.module.enc_normalizer.denormalize(src_latent)
    else:
        src_latent = None
        src_mask = None
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        src_mask = src["attention_mask"]
        # batch["text_features"] = noise_injection(batch["text_features"], 0.016)
        text_features = decoder.linear(batch['text_features']).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
        prefix = decoder.prefix_const.unsqueeze(dim=0).expand(batch['text_features'].shape[0], *decoder.prefix_const.shape)  # (b, prefix_length, d_model)

        rt_captions = decoder.rt_linear(batch['rt_captions'].to(device))
        att_gt_n_rt = decoder.crossatt(text_features, rt_captions)

        inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
        text_features = decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
        text_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
        src_latent = torch.concat((text_features, src_latent), dim = 1)
        src_mask = torch.concat((text_features_mask, src_mask), dim = 1)          
        logits = decoder(latent, cond_x=src_latent, cond_mask=src_mask)

    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def save_checkpoint(model, config):
    os.makedirs(config.training.checkpoints_folder, exist_ok=True)

    model.eval()

    torch.save(
        {
            "decoder": model.state_dict(),
        },
        config.decoder.decoder_path
    )

    print(f"Save model to: {config.decoder.decoder_path}")


def train(config, encoder, decoder, tokenizer):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    batch_size = config.decoder.batch_size

    train_dataset, valid_dataset = get_datasets(config=config)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.decoder.lr,
        weight_decay=config.decoder.weight_decay,
        betas=config.decoder.betas,
    )

    step = 0
    for _ in range(config.decoder.epochs): 
        train_loader, valid_loader = get_loaders(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size
        )

        decoder.train()
        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                decoder=decoder,
                config=config,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=config.decoder.max_norm
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

        
        # decoder.eval()
        # with torch.no_grad():
        #     total_loss = 0.
        #     total_acc = 0.
        #     total_num = 0.
            
        #     for batch in tqdm(valid_loader):
        #         loss, acc = loss_step(
        #             batch=batch,
        #             tokenizer=tokenizer,
        #             encoder=encoder,
        #             decoder=decoder,
        #             config=config,
        #             eval=True
        #         )
        #         total_loss += loss * len(batch['text_trg'])
        #         total_acc += acc * len(batch['text_trg'])
        #         total_num += len(batch['text_trg'])

        #     total_loss /= total_num
        #     total_acc /= total_num

        #     wandb.log({f'valid loss': total_loss.item()}, step=step)
        #     wandb.log({f'valid accuracy': total_acc.item()}, step=step)
        save_checkpoint(decoder, config)


def main():
    args = parse()
    
    config = create_config(args)
    if not config.emb:
        enc_normalizer = EncNormalizer(
            enc_mean_path=config.data.enc_gen_mean,
            enc_std_path=config.data.enc_gen_std,
        )
    else:
        enc_normalizer = None
    encoder = Encoder(
        config.model.encoder_link,
        enc_normalizer=enc_normalizer,
        is_change_sp_tokens=True,
        emb=config.emb
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)

    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = Decoder(decoder_config=config.decoder, diffusion_config=config.se_config).train().cuda()

    wandb.init(project=config.project_name, name=config.decoder.name, mode="offline")
    train(config, encoder, decoder, tokenizer)

def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std)
    else:
        x = x + (torch.randn(x.shape, device=device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=1)

def get_uniform_ball_noise(input_shape, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape, device="cuda:0")  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
    u = torch.rand(input_shape[0], device="cuda:0")  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball

main()
