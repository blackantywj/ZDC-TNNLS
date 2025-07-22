from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
from itertools import chain
import argparse
from huggingface_hub import HfApi
import nltk
import os
import pickle

def download_wikipedia(dataset_path):
    # for private dataset 
    # huggingface-cli login

    dt = load_dataset("bigscience-data/roots_en_wikipedia")
    dt = dt["train"]
    dt = dt.remove_columns("meta")

    # Split Articles by \n\n
    def split(batch):
        result = []
        for text in batch["text"]:
            texts = text.split("\n\n")
            result.append(texts)
        result = list(chain(*result))
        return {"text": result}
    
    dt = dt.map(
        split,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
    )

    # Filter small texts
    min_symbols = 600
    dt = dt.filter(lambda b: len(b["text"]) >= min_symbols, num_proc=30)

    # Split into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def split_into_sents(batch):
        result = []
        for text in batch["text"]:
            texts = tokenizer.tokenize(text)
            result.append(texts)
        result = list(chain(*result))
        return {"text": result}
    
    sent_dt = dt.map(
        split_into_sents,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
    )

    def join_sents(batch):
        result = []
        cur_text = ''
        for text in batch["text"]:
            if len(cur_text.split()) + len(text.split()) < 128 / 2:
                cur_text += ' ' + text
            else:
                result.append(cur_text)
                cur_text = text
        
        return {"text": result}

    joined_dt = sent_dt.map(
        join_sents,
        batched=True,
        num_proc=30,
        desc="Dataset join",
        batch_size=100000,
    )

    dt = joined_dt.train_test_split(test_size=0.002, seed=0)
    dt.save_to_disk(
        dataset_path,
        num_shards={'train': 20, 'test': 1}
    )


def download_qqp(dataset_path):
    dt = load_dataset("glue", "qqp")
    dt = dt.filter(lambda x: x["label"] == 1)
    dt = dt.remove_columns(["label", "idx"])
    dt = concatenate_datasets([dt["train"], dt["validation"]])
    
    dt = dt.train_test_split(test_size=0.2, seed=0)
    dt_train = dt["train"]
    dt = dt["test"].train_test_split(test_size=0.5, seed=0)

    dt = DatasetDict(
        {
            "train": dt_train,
            "validation": dt["train"],
            "test": dt["test"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_xsum(dataset_path):
    dt = load_dataset("EdinburghNLP/xsum")
    dt.save_to_disk(dataset_path)


def download_wiki_auto(dataset_path):
    dt = load_dataset("GEM/wiki_auto_asset_turk")
    dt = dt.remove_columns(["gem_id", "gem_parent_id"])
    
    dt = DatasetDict(
        {
            "train": dt["train"],
            "validation": dt["validation"],
            "test": dt["test_asset"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_squad(dataset_path):
    def make_batch(batch):
        new_batch = {
            "source": [],
            "target": [],
        }

        for context, answer, target in zip(batch["context"], batch["answers"], batch["target"]):
            if answer["text"]:
                new_batch["source"].append(f"Context: {context}. Answer: {answer['text'][0]}.")
                new_batch["target"].append(target)
        return new_batch
    
    dt = load_dataset("GEM/squad_v2")
    dt = dt.map(
        make_batch,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
        remove_columns=dt["train"].column_names
    )
    dt.save_to_disk(dataset_path)
    

def download_rocstory(dataset_path):
    def preprocess(batch):
        targets = []
        size = len(batch["storyid"])
        for i in range(size):
            text = " ".join([batch[f"sentence{k}"][i] for k in range(1, 6)])
            targets.append(text)
        return {"target": targets}

    dt = load_dataset("wza/roc_stories")
    dt = dt["train"]
    dt = dt.map(
        preprocess,
        batched=True,
        num_proc=30,
        desc="Loading...",
        remove_columns=dt.column_names,
    )
    dt = dt.train_test_split(test_size=10000, seed=0)
    dt.save_to_disk(dataset_path)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json 
def select_tfidf_negatives_dict(sentences, ratios=(0.30, 0.40, 0.50)):
    """
    基于 TF–IDF 选择负样本，按比例返回无重复句子的字典。

    参数:
    - sentences: List[str]，所有句子列表
    - ratios: Tuple[float]，要抽取的比例，如 (0.05, 0.10, 0.15)

    返回:
    - dict: {'5%': [...], '10%': [...], '15%': [...]}，对应比例下的负样本句子列表，保证唯一且无重复。
    """
    # 1. 计算 TF–IDF 矩阵
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None,
                                 min_df=1, max_df=1.0)
    X = vectorizer.fit_transform(sentences)  # (n_sentences, n_features)

    # 2. 汇总每句 TF–IDF 得分（sum）：句子自信息量近似
    scores = X.sum(axis=1).A1  # nd-array

    # 3. 根据得分排序（升序：低分即低信息量）
    idx_sorted = np.argsort(scores)
    n = len(sentences)

    # 4. 按比例选取并去重
    negatives = {}
    for ratio in ratios:
        k = max(1, int(n * ratio))  # 至少选一个
        selected_idxs = idx_sorted[:k]
        # 保持原句子顺序并去重
        seen = set()
        unique_sents = []
        for idx in selected_idxs:
            sent = sentences[idx]
            if sent not in seen:
                seen.add(sent)
                unique_sents.append(sent)
        negatives[f'{int(ratio*100)}%'] = unique_sents
    return negatives

def download_mscoco(dataset_path):
    # 加载整体 COCO-caption 数据集
    ds_train = load_dataset("yerevann/coco-karpathy", split="train+restval")
    ds_val = load_dataset("yerevann/coco-karpathy", split="validation")
    ds_test = load_dataset("yerevann/coco-karpathy", split="test")
    with open("/home/cumt/workspace/IFCap/annotations/coco/coco_with_entities.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    # ds_train['sentences'] = select_tfidf_negatives_dict(ds_train['sentences'])["50%"]
    src_tag_dict = {}
    neg_samples_50_ids = json.load(open("/home/cumt/workspace/tencdm_mscoco/neg_samples_50_ids.json"))
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = [] 
    for temp_captions in ds_train['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_train = ds_train.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_mscoco/coco_with_entities_val.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_val['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_val = ds_val.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_mscoco/coco_with_entities_test.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_test['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_test = ds_test.add_column("entities", entities)

    dt = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )
    # 如果只需文本，可提取 captions 列
    # texts = ds["sentences"]    # 每条是 captions 列表，若要扁平化可进一步处理
    dt.save_to_disk(dataset_path)

def download_nocaps(dataset_path):
    # 加载整体 COCO-caption 数据集
    ds_validation = load_dataset("lmms-lab/NoCaps", split="validation")

    # with open("/home/cumt/wjworkspace/tencdm/coco_with_entities_val.pickle", 'rb') as file:
    #     src_tag = pickle.load(file)
    # src_tag_dict = {}
    # for item in src_tag:
    #     src_tag_dict[item[1]] = item[0]
    # # 或者加载 Karpathy-split 纯文本版本
    # # ds = load_dataset("yerevann/coco-karpathy", split="train")
    # punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    # entities = []
    # for temp_captions in ds_validation['sentences']:
    #     tmplist = []
    #     for caption in temp_captions:                  # caption
    #         caption = caption.strip()                  # removing space at the end of the caption
    #         if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
    #             caption = caption.lower()
    #         caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
    #         if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
    #             caption += '.'
    #         tmplist.append(src_tag_dict[caption])
    #     entities.append(tmplist) 
    # ds_val = ds_val.add_column("entities", entities)

    dt = DatasetDict(
        {
            "validation": ds_validation,
        }
    )
    # 如果只需文本，可提取 captions 列
    # texts = ds["sentences"]    # 每条是 captions 列表，若要扁平化可进一步处理
    dt.save_to_disk(dataset_path)

def download_mscoco_ng(dataset_path):
    # 加载整体 COCO-caption 数据集
    ds_train = load_dataset("yerevann/coco-karpathy", split="train+restval")
    ds_val = load_dataset("yerevann/coco-karpathy", split="validation")
    ds_test = load_dataset("yerevann/coco-karpathy", split="test")
    with open("/home/cumt/workspace/IFCap/annotations/coco/coco_with_entities.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    # ds_train['sentences'] = select_tfidf_negatives_dict(ds_train['sentences'])["50%"]
    src_tag_dict = {}
    with open("gram_neg_samples_30_ids.json", "r", encoding="utf-8") as f:
        neg_ids = set(json.load(f))
# 3. 定义一个清理函数：对于每个 example，过滤 out_neg=True 的条目
    def remove_neg_sentences(example):
        sentids = example["sentids"]
        sentences = example["sentences"]
        # 保留那些 ID 不在 neg_ids 里的索引
        keep_indices = [i for i, sid in enumerate(sentids) if str(sid) in neg_ids]
        
        # 用于 map 返回的新字段
        example["sentids"]   = [sentids[i]   for i in keep_indices]
        example["sentences"] = [sentences[i] for i in keep_indices]
        return example

    # 4. 用 map 做清理
    ds_cleaned = ds_train.map(
        remove_neg_sentences,
        batched=False,
        remove_columns=[]  # 不删其它字段
    )

    # 5. （可选）如果你希望把所有句子都被删光的样本也丢掉：
    ds_filtered = ds_cleaned.filter(
        lambda ex: len(ex["sentids"]) > 0
    )

    # 6. 查看数量对比
    print("原始样本数：", len(ds_train))
    print("清理后（保留空样本）：", len(ds_cleaned))
    print("再删空样本之后：", len(ds_filtered))
    ds_train = ds_filtered
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_train['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_train = ds_train.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_mscoco/coco_with_entities_val.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_val['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_val = ds_val.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_mscoco/coco_with_entities_test.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_test['sentences']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_test = ds_test.add_column("entities", entities)

    dt = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )
    # 如果只需文本，可提取 captions 列
    # texts = ds["sentences"]    # 每条是 captions 列表，若要扁平化可进一步处理
    dt.save_to_disk(dataset_path)

def download_flickr30k_ng(dataset_path):
    # 先把唯一的 split 全部读进来
    ds = load_dataset("nlphuji/flickr30k", split="test")
    
    # 按内部字段再拆分 '2637904605.jpg' in ds_train['filename']
    ds_train = ds.filter(lambda x: x["split"] == "train")
    ds_val   = ds.filter(lambda x: x["split"] == "val")
    ds_test  = ds.filter(lambda x: x["split"] == "test")
    with open("/home/cumt/workspace/IFCap/annotations/flickr30k/flickr30k_with_entities.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    # ds_train['sentences'] = select_tfidf_negatives_dict(ds_train['sentences'])["50%"]
    src_tag_dict = {}
    with open("flickr30k_neg_samples_10_ids.json", "r", encoding="utf-8") as f:
        neg_ids = set(json.load(f))
# 3. 定义一个清理函数：对于每个 example，过滤 out_neg=True 的条目
    def remove_neg_sentences(example):
        sentids = example["sentids"]
        sentences = example["caption"]
        # 保留那些 ID 不在 neg_ids 里的索引
        keep_indices = [i for i, sid in enumerate(sentids) if str(sid) in neg_ids]
        
        # 用于 map 返回的新字段
        example["sentids"]   = [sentids[i]   for i in keep_indices]
        example["caption"] = [sentences[i] for i in keep_indices]
        return example

    # 4. 用 map 做清理
    ds_cleaned = ds_train.map(
        remove_neg_sentences,
        batched=False,
        remove_columns=[]  # 不删其它字段
    )

    # 5. （可选）如果你希望把所有句子都被删光的样本也丢掉：
    ds_filtered = ds_cleaned.filter(
        lambda ex: len(ex["sentids"]) > 0
    )

    # 6. 查看数量对比
    print("原始样本数：", len(ds_train))
    print("清理后（保留空样本）：", len(ds_cleaned))
    print("再删空样本之后：", len(ds_filtered))
    ds_train = ds_filtered
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_train['caption']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_train = ds_train.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_coco_text_fea/flickr30k_with_entities_val.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_val['caption']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_val = ds_val.add_column("entities", entities)

    with open("/home/cumt/workspace/tencdm_coco_text_fea/flickr30k_with_entities_test.pickle", 'rb') as file:
        src_tag = pickle.load(file)
    src_tag_dict = {}
    for item in src_tag:
        src_tag_dict[item[1]] = item[0]
    # 或者加载 Karpathy-split 纯文本版本
    # ds = load_dataset("yerevann/coco-karpathy", split="train")
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    entities = []
    for temp_captions in ds_test['caption']:
        tmplist = []
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            tmplist.append(src_tag_dict[caption])
        entities.append(tmplist) 
    ds_test = ds_test.add_column("entities", entities)

    dt = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )
    # 如果只需文本，可提取 captions 列
    # texts = ds["sentences"]    # 每条是 captions 列表，若要扁平化可进一步处理
    dt.save_to_disk(dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset arguments")
    parser.add_argument(
        "--dataset_name", type=str, default=None, 
        choices=[
            "rocstories", 
            "wikipedia", 
            "qqp", "xsum", "wiki_auto", "nocaps", "flickr30k", 
            "flickr30k_ng_5", "flickr30k_ng_5_re", "flickr30k_ng_3", "flickr30k_ng_1", 
            'mscoco','mscoco_ng','mscoco_ng_7', "mscoco_ng_3", "mscoco_ng_4", 
            "mscoco_ng_2", "mscoco_ng_8", "mscoco_ng_3_2g"
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset_path", type=str, default=''.join([os.getcwd(), '/datasets/']),
        required=False,
    )
    
    args = parser.parse_args()

    if args.dataset_name == "rocstories":
        download_rocstory(args.dataset_path + args.dataset_name)
    if args.dataset_name == "wikipedia":
        download_wikipedia(args.dataset_path + args.dataset_name)
    if args.dataset_name == "qqp":
        download_qqp(args.dataset_path + args.dataset_name)
    if args.dataset_name == "xsum":
        download_xsum(args.dataset_path + args.dataset_name)
    if args.dataset_name == "wiki_auto":
        download_wiki_auto(args.dataset_path + args.dataset_name)
    if args.dataset_name == "mscoco":
        download_mscoco(args.dataset_path + args.dataset_name)
    # if args.dataset_name == "flickr30k":
    #     download_mscoco(args.dataset_path + args.dataset_name)
    if "flickr30k_ng" in args.dataset_name:
        download_flickr30k_ng(args.dataset_path + args.dataset_name)
    if "mscoco_ng" in args.dataset_name:
        download_mscoco_ng(args.dataset_path + args.dataset_name)
    if "nocaps" in args.dataset_name:
        download_nocaps(args.dataset_path + args.dataset_name)
    # if args.dataset_name == "mscoco_ng_7":
    #     download_mscoco_ng(args.dataset_path + args.dataset_name)
    # if args.dataset_name == "mscoco_ng_3":
    #     download_mscoco_ng(args.dataset_path + args.dataset_name)