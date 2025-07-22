import os
import torch
import wandb
import numpy as np
import argparse
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from utils.util import dict_to_cuda
from model.decoder_unt import Decoder
from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse
import torch.distributions as dist
from torch.distributions.categorical import Categorical
import torch
import clip, math
import pickle

os.environ["WANDB_API_KEY"] = '37f1c58d82d00a91be324ac7a587c9996718f770' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device, jit=False)  # load model & ignore image preprocess

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Initialize the plot
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Assume you have lists storing the loss values over time
loss_pred_values = []
loss_hid_values = []
loss_att_values = []

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

def loss_step(batch, tokenizer, encoder, decoder, t_decoder, config, eval=False, step = None):
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
        batch["text_features"] = noise_injection(batch["text_features"], 0.016)
        if config.decoder.ssf:
            text_features = decoder.linear(batch['text_features']).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
            prefix = decoder.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *decoder.prefix_const.shape)  # (b, prefix_length, d_model)

            rt_captions = decoder.rt_linear(batch['rt_captions'].to(device))
            att_gt_n_rt = decoder.crossatt(text_features, rt_captions)

            inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            
            text_features = decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            text_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")  
            src_latent_s = torch.concat((text_features, src_latent), dim = 1)
            src_mask = torch.concat((text_features_mask, src_mask), dim = 1)          
            logits, all_hidden_states, all_self_attentions = decoder(latent, cond_x=src_latent_s, cond_mask=src_mask, output_hidden_states = True, output_attentions = True)
          
        else:
            text_features = decoder.linear(batch['text_features']).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
            prefix = decoder.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *decoder.prefix_const.shape)  # (b, prefix_length, d_model)

            # rt_captions = decoder.rt_linear(batch['rt_captions'].to(device))
            # att_gt_n_rt = decoder.crossatt(text_features, rt_captions)
            inputs = torch.cat((text_features, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            text_features = decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            text_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")  
            src_latent_s = torch.concat((text_features, src_latent), dim = 1)
            src_mask = torch.concat((text_features_mask, src_mask), dim = 1)                    
            logits, all_hidden_states, all_self_attentions = decoder(latent, cond_x=src_latent_s, cond_mask=src_mask, output_hidden_states = True, output_attentions = True)

    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    loss = reconstruction_loss(targets, logits, mask=None)

    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)    

    loss_pred = 0.0
    
    tmp = config.decoder.temperature
    
    if config.decoder.negative_train:
        with torch.no_grad():
            src_mask = src["attention_mask"]
            
            text_features = t_decoder.linear(batch['text_features'].to(torch.float32)).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
            prefix = t_decoder.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *t_decoder.prefix_const.shape)  # (b, prefix_length, d_model)

            rt_captions = t_decoder.rt_linear(batch['rt_captions'].to(device))
            att_gt_n_rt = t_decoder.crossatt(text_features, rt_captions)

            inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
            text_features = t_decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
            text_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
            src_latent = torch.concat((text_features, src_latent), dim = 1)
            src_mask = torch.concat((text_features_mask, src_mask), dim = 1)          
            teacher_logits, teacher_all_hidden_states, teacher_all_self_attentions = t_decoder(latent, cond_x=src_latent, cond_mask=src_mask, output_hidden_states = True, output_attentions = True)        
            p_t = F.softmax(teacher_logits / tmp, dim=-1)
            
        if config.decoder.add_pred:
            # —— 1) 预测层 软反可能性 L_pred —— 
            p_s = F.softmax(logits / tmp, dim=-1)
            loss_pred = -(p_t * torch.log(1 - p_s + 1e-12)).sum()
            # Calculate intermediate values
            log_term = torch.log(1 - p_s + 1e-12)
            print("log(1 - p_s + 1e-12):", log_term[0])
            # Compute loss component-wise
            loss_component = p_t * log_term
            print("loss_component (p_t * log_term):", loss_component[0])
            loss_pred = loss_pred * (tmp ** 2)    

        loss_hid = 0.0
        loss_att = 0.0
        if config.decoder.add_hid:
            # 跳过 embedding 层，只对后续 hidden 层做负蒸馏
            hidden_layers = list(zip(all_hidden_states[1:], teacher_all_hidden_states[1:]))
            n_layers = len(hidden_layers)
            for hs, ht in hidden_layers:
                # 对最后一维做 softmax 以获得概率分布
                # P = F.softmax(ht / tmp, dim=-1)
                # Q = F.softmax(hs / tmp, dim=-1) .log(), ht, reduction='none').sum(-1)
                ht = Categorical(logits=ht)
                hs = Categorical(logits=hs)
                kl = dist.kl_divergence(ht, hs)
                # 负教师蒸馏保留 exp：exp(-kl) 将鼓励 student 远离 teacher 的输出模式
                loss_hid += torch.exp(-kl).mean()  # mean over batch×seq_len
            # 平均层数
            loss_hid = (loss_hid / n_layers)

        if config.decoder.add_att:
            att_layers = list(zip(all_self_attentions, teacher_all_self_attentions))
            n_att = len(att_layers)
            for as_, at_ in att_layers:
                # Attention maps are [B, heads, L, L]
                # at_ = F.softmax(at_ / tmp, dim=-1)
                # as_ = F.softmax(as_ / tmp, dim=-1) F.kl_div(at_.log(), as_, reduction='none').sum(-1)  # sum over last dim 
                at_ = Categorical(logits=at_)
                as_ = Categorical(logits=as_)
                kl = dist.kl_divergence(at_, as_)
                loss_att += torch.exp(-kl).mean()  # mean over B×heads×L
            loss_att = (loss_att / n_att)
        # —— 总 UNKT 损失 —— 
        print(f'loss_pred: {loss_pred}, loss_hid: {loss_hid}, loss_att: {loss_att}', '\n')

        # Append the losses to their respective lists
        loss_pred_values.append(loss_pred.cpu().detach().numpy())
        loss_hid_values.append(loss_hid.cpu().detach().numpy())
        loss_att_values.append(loss_att.cpu().detach().numpy())
        # Clear the previous output and plot the updated loss values
        clear_output(wait=True)

        # Plot for loss_pred
        axs[0].cla()  # Clear the previous plot for loss_pred
        axs[0].plot(loss_pred_values, label='loss_pred', color='blue')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot for loss_hid
        axs[1].cla()  # Clear the previous plot for loss_hid
        axs[1].plot(loss_hid_values, label='loss_hid', color='green')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        # Plot for loss_att
        axs[2].cla()  # Clear the previous plot for loss_att
        axs[2].plot(loss_att_values, label='loss_att', color='red')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')
        axs[2].legend()

        # Adjust layout and display the updated figure
        plt.tight_layout()
        # Save the plot every 10 epochs (or any other interval)
        if step % 1000 == 0:
            plot_filename = f"loss_plot/loss_plot_epoch_{step}.png"
            plt.savefig(plot_filename)
            print(f"Plot saved: {plot_filename}")
        loss_pred = loss_pred + loss_hid + loss_att
    pickle_filename = './loss_values.pkl'

    # Save the lists to a pickle file
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'loss_pred': loss_pred_values,
            'loss_hid': loss_hid_values,
            'loss_att': loss_att_values
        }, f)        
    return loss, loss_pred, acc


def save_checkpoint(model, config):
    os.makedirs(config.training.checkpoints_folder, exist_ok=True)

    model.eval()
    
    if config.decoder.negative_train:
        originname = config.decoder.decoder_path.split('/')[-1].split('.')[0]
        config.decoder.decoder_path = f"datasets/{config.decoder.decoder_path.split('/')[-2]}/{originname}_ssf_{config.decoder.ssf}_tmp_{config.decoder.temperature}_ADD_HID_{str(config.decoder.add_hid)}_ADD_ATT_{str(config.decoder.add_att)}_epoch_{config.decoder.epochs}_nl_{config.decoder.num_hidden_layers}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_centerpos_{config.decoder.centerpos}_{config.decoder.percent}.pth"
    elif {config.decoder.decoder_path.split('/')[-2]} in ['flickr30k', 'mscoco']:
        originname = config.decoder.decoder_path.split('/')[-1].split('.')[0]
        config.decoder.decoder_path = f"datasets/{config.decoder.decoder_path.split('/')[-2]}/{originname}_ssf_{config.decoder.ssf}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_nl_{config.decoder.num_hidden_layers}_epoch_{config.decoder.epochs}.pth"    
    else:
        originname = config.decoder.decoder_path.split('/')[-1].split('.')[0]
        config.decoder.decoder_path = f"datasets/{config.decoder.decoder_path.split('/')[-2]}/{originname}_ssf_{config.decoder.ssf}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_nl_{config.decoder.num_hidden_layers}.pth"    
    torch.save(
        {
            "decoder": model.state_dict(),
        },
        config.decoder.decoder_path
    )
    print(f"Save model to: {config.decoder.decoder_path}")

def progressive_lambda(step, T):
    """Compute the progressive lambda value based on cosine decay."""
    return 0.5 * (1 + math.cos(step / T * math.pi / 2))

def lambda_cosine_decay(step, T, start=1.0, end=2/3):
    """
    Cosine decay from `start` down to `end` over `T` steps.
    
    At step=0 → returns start.
    At step=T → returns end.
    """
    # 标准余弦：cos(0)=1 → cos(pi) = -1
    cosine = math.cos(math.pi * step / T)
    # 归一化到 [0,1]：0.5*(1+cosine) 从 1 衰减到 0
    decay = 0.5 * (1 + cosine)
    # 映射到 [start, end]
    return end + (start - end) * decay

import math

def alpha_schedule(s, T=4428, lam=4.0, beta=None, gamma=None):
    """
    进度系数 α(s) 计算函数：
    
    α(s) = lam * exp(-z) / ( (exp(-z) + 1)^2 )
    z    = beta * (s - gamma)
    
    参数：
      s     -- 当前训练步数（0 ≤ s ≤ T）
      T     -- 总训练步数（默认 4428）
      lam   -- 峰值控制因子（默认 4.0）
      beta  -- 展开/收缩速率，若为 None 则设为 12/T
      gamma -- 峰值位置，若为 None 则设为 T/2
    
    返回：
      0 ≤ α(s) ≤ lam 的平滑先升后降曲线值
    """
    # 默认参数
    if beta is None:
        beta = 6.0 / T
    if gamma is None:
        gamma = T / 2.0
    
    # 计算 z(s)
    z = beta * (s - gamma)
    # 计算 α
    return lam * math.exp(-z) / ((math.exp(-z) + 1) ** 2)

def progressive_alpha(step: int,
                      total_steps: int,
                      center: float = None) -> float:
    """
    计算平滑的先升后降系数 alpha(step)，曲线形状与 sigmoid 导数相同。
    
    参数：
      - step: 当前训练步数（0 <= step <= total_steps）
      - total_steps: 总训练步数
      - peak: 曲线的最大高度（即峰值），默认 1.0
      - beta_factor: 控制升降的陡峭程度。较大值→曲线更尖锐，较小值→更平缓。
                     默认 12.0，相当于 z 从约 -6 到 +6 跨度。
      - center: 峰值出现的位置（步数）。默认 None 时取 total_steps/2。
    
    返回：
      - 对应 step 时刻的 alpha，范围在 (0, peak] 之间。
    """
    # 峰值位置
    if center is None:
        center = total_steps / 2
    # 归一化后的 beta
    # beta = beta_factor / total_steps
    # 计算 z
    if step < center:
        ratio = step / center
        # sigmoid 导数形式
        return 0.5 * (1 - math.cos(math.pi * ratio))
    else:
        """Compute the progressive lambda value based on cosine decay."""
        return 0.5 * (1 + math.cos((step - center) / (total_steps - center) * math.pi / 2))

def train(config, encoder, decoder, t_decoder, tokenizer):
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
    
    train_loader, valid_loader = get_loaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size
    )
    T = (config.decoder.epochs * len(train_loader))
    decoder.train()
    
    for epoch in range(config.decoder.epochs): 
        for batch_idx, batch in tqdm(enumerate(train_loader)):

            loss, ngloss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                decoder=decoder,
                t_decoder = t_decoder,
                config=config,
                step = epoch * len(train_loader) + batch_idx
            )

            if config.decoder.negative_train:
                step = epoch * len(train_loader) + batch_idx
                alpha = progressive_alpha(step, T, center = T / config.decoder.centerpos)
                print(f'alpha: {alpha}', '\n')
                loss = (1 - alpha) * loss + alpha * ngloss       

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

    save_checkpoint(decoder, config)
    # Define the pickle filename

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
    if "gpt2" in encoder.encoder_link:
        tokenizer.pad_token = tokenizer.eos_token  
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = Decoder(decoder_config=config.decoder, diffusion_config=config.se_config).train().cuda()
    if config.decoder.negative_train:
        t_decoder = Decoder(decoder_config=config.decoder, diffusion_config=config.se_config).eval().cuda()
        t_decoder.load_state_dict(torch.load(f"/home/cumt/workspace/tencdm_mscoco/datasets/{config.data.datasets.datasets_list[0]}_ng_{config.decoder.percent}/decoder-{config.model.encoder_name.lower()}-128-transformer-conditional_ssf_{config.decoder.ssf}_bs_{config.decoder.batch_size}_lr_{config.decoder.lr}_nl_{config.decoder.num_hidden_layers}.pth")['decoder'])
    else:
        t_decoder = None
    wandb.init(project=config.project_name, name=config.decoder.name)
    train(config, encoder, decoder, t_decoder, tokenizer)

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

def hard_neg_from_student(teacher_logits, student_logits, tmp, m=30, n=5):
    """
    teacher_logits, student_logits: [B, T, V]
    返回 p_t_neg: [B, T, V]
    """
    # 教师 & 学生分布
    p_t = F.softmax(teacher_logits / tmp, dim=-1)
    p_s = F.softmax(student_logits / tmp, dim=-1)

    # top-m 索引
    _, topk_t = torch.topk(p_t, m, dim=-1)  # 教师 top-m
    _, topk_s = torch.topk(p_s, m, dim=-1)  # 学生 top-m

    # 对每个位置找交集或并集 minus teacher 最优
    B, T, _ = p_t.shape
    p_neg = torch.zeros_like(p_t)
    for b in range(B):
        for t in range(T):
            set_t = set(topk_t[b, t].tolist())
            set_s = set(topk_s[b, t].tolist())
            # 候选 hard-neg: 学生也喜欢但不是教师最强选
            candidates = list(set_s & set_t - {topk_t[b, t, 0].item()})
            if len(candidates) == 0:
                continue
            # 按教师概率从 high->low 取前 n 个
            c_probs = [(idx, p_t[b, t, idx].item()) for idx in candidates]
            c_probs.sort(key=lambda x: x[1], reverse=True)
            selected = [idx for idx, _ in c_probs[:n]]
            # 用教师概率作为负分布的权重
            weights = torch.tensor([p_t[b, t, idx] for idx in selected],
                                   device=p_t.device)
            weights = weights / weights.sum()
            for idx, w in zip(selected, weights):
                p_neg[b, t, idx] = w

    # 归一化防止分母为 0
    denom = p_neg.sum(-1, keepdim=True).clamp_min(1e-12)
    p_neg = p_neg / denom
    return p_neg


main()


#    if config.decoder.negative_train:
#         with torch.no_grad():
#             src_mask = src["attention_mask"]
            
#             text_features = t_decoder.linear(batch['text_features'].to(torch.float32)).view(batch['text_features'].shape[0], 10, -1)  # (b, clip_project_length, d_model)
#             prefix = t_decoder.prefix_const.unsqueeze(dim=0).expand(text_features.shape[0], *t_decoder.prefix_const.shape)  # (b, prefix_length, d_model)

#             rt_captions = t_decoder.rt_linear(batch['rt_captions'].to(device))
#             att_gt_n_rt = t_decoder.crossatt(text_features, rt_captions)

#             inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
#             text_features = t_decoder.transformer(inputs)[:, 10:, :] # (b, prefix_length, d_model)        
#             text_features_mask = torch.ones(text_features.shape[0], 10, dtype=torch.int64, device="cuda:0")        
#             src_latent = torch.concat((text_features, src_latent), dim = 1)
#             src_mask = torch.concat((text_features_mask, src_mask), dim = 1)          
#             teacher_logits, teacher_all_hidden_states, teacher_all_self_attentions = t_decoder(latent, cond_x=src_latent, cond_mask=src_mask, output_hidden_states = True, output_attentions = True)        
#             p_t = F.softmax(teacher_logits / tmp, dim=-1)
            
#         if config.decoder.add_pred:
#             # —— 1) 预测层 软反可能性 Lpred —— 
#             p_s = F.softmax(logits / tmp, dim=-1)
#             loss_pred = -(p_t * torch.log(1 - p_s + 1e-12)).sum()
#             loss_pred = loss_pred * (tmp ** 2)    

#         loss_hid = 0.0
#         loss_att = 0.0
#         if config.decoder.add_hid:
#             # 跳过 embedding 层，只对后续 hidden 层做负蒸馏
#             hidden_layers = list(zip(all_hidden_states[1:], teacher_all_hidden_states[1:]))
#             n_layers = len(hidden_layers)
#             for hs, ht in hidden_layers:
#                 # 对最后一维做 softmax 以获得概率分布
#                 P = F.softmax(ht / tmp, dim=-1)
#                 Q = F.softmax(hs / tmp, dim=-1)
#                 # 计算 P||Q 的 KL divergence: sum P * (log P - log Q)
#                 kl = F.kl_div(P.log(), Q, reduction='none').sum(-1)
#                 # 负教师蒸馏保留 exp：exp(-kl) 将鼓励 student 远离 teacher 的输出模式
#                 loss_hid += torch.exp(-kl).mean()  # mean over batch×seq_len
#             # 平均层数并还原温度缩放
#             loss_hid = (loss_hid / n_layers) * (tmp ** 2)

#         if config.decoder.add_att:
#             att_layers = list(zip(all_self_attentions, teacher_all_self_attentions))
#             n_att = len(att_layers)
#             for as_, at_ in att_layers:
#                 # Attention maps are [B, heads, L, L]
#                 at_ = F.softmax(at_ / tmp, dim=-1)
#                 as_ = F.softmax(as_ / tmp, dim=-1)
#                 kl = F.kl_div(at_.log(), as_, reduction='none').sum(-1)  # sum over last dim
#                 loss_att += torch.exp(-kl).mean()  # mean over B×heads×L
#             loss_att = (loss_att / n_att) * (tmp ** 2)
#         # —— 总 UNKT 损失 —— 
#         loss_pred = loss_pred + loss_hid + loss_att