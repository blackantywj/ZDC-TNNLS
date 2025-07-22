import functools
from collections import defaultdict
import numpy as np
import pickle, json
from data.util import parse_entities, load_stopwords, compose_discrete_prompts
from data.load_annotations import load_entities_text
import torch, clip

name_of_objects_vocabs = "visual_genome_entities"
path_of_objects_vocabs = "/home/cumt/workspace/tencdm_mscoco/all_objects_attributes_relationships.pickle"
objects_vocabs = load_entities_text(name_of_objects_vocabs, path_of_objects_vocabs, all_entities = False)
people_vocabs = ['people', 'person', 'man', 'men', 'woman', 'women', 'adult','boy', 'girl', 'kid', 'children', 'child', 'baby', 'guy', 'player', 'male', 'female', 'worker']
stopwords = load_stopwords()

with open("/home/cumt/workspace/tencdm_mscoco/all_coco_with_entities_cocoid_imgfeature.pickle", 'rb') as file:
    alldata_coco = pickle.load(file)

cap_feat_coco = {c.rstrip():f.squeeze(0).cpu() for e, c, f, g, h in alldata_coco.values()}

rt_caps_coco = json.load(open("/home/cumt/workspace/tencdm_mscoco/all_coco_train_woclass.json"))
for key in rt_caps_coco.keys():
    rt_caps_coco[key] = rt_caps_coco[key][:5]

with open("/home/cumt/workspace/tencdm_mscoco/all_flickr30k_with_entities_cocoid_imgfeature.pickle", 'rb') as file:
    alldata_flickr30k = pickle.load(file)

cap_feat_flickr30k = {c.rstrip():f.squeeze(0).cpu() for e, c, f, g, h in alldata_flickr30k.values()}

rt_caps_flickr30k = json.load(open("/home/cumt/workspace/tencdm_mscoco/all_flickr30k_train.json"))
for key in rt_caps_flickr30k.keys():
    rt_caps_flickr30k[key] = rt_caps_flickr30k[key][:5]

with open("/home/cumt/workspace/tencdm_mscoco/test_nocaps_with_entities_cocoid_imgfeature_new.pickle", 'rb') as file:
    alldata_nocaps = pickle.load(file)

rt_caps_nocaps = json.load(open("/home/cumt/workspace/tencdm_mscoco/nocaps_test.json"))
for key in rt_caps_nocaps.keys():
    rt_caps_nocaps[key] = rt_caps_nocaps[key][:5]

device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, _ = clip.load("ViT-B/32", device=device, jit=False) 
token = clip.tokenize(["", "", "", "", "",]).to(device)
with torch.no_grad():
    replace = clipmodel.encode_text(token)

def batch_preprocessing_mscoco(batch, dataset_name, split, swap_cfg_coef):
    if dataset_name in ["mscoco"] or "mscoco_ng" in dataset_name:
        text_src = []
        text_tag = []
        img_features_list = []
        text_tag = []
        cocoid_list = []
        rt_captions = []
        if split == 'train' :
            for sentids, cocoid in zip(batch['sentids'], batch['cocoid']):
                for idx in sentids: 
                    temp_detected_entities, temp_caption, temp_clip_features, cocoid, temp_img_fearures = alldata_coco[idx]
                    src_st = parse_entities([temp_detected_entities], stopwords, people_vocabs, objects_vocabs)[0]
                    text_src.append(src_st)
                    text_tag.append(temp_caption)
                    rt_captions.append(torch.stack(list(map(cap_feat_coco.get, rt_caps_coco[temp_caption.rstrip()]))))

            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "rt_captions": rt_captions
            }

        else:
            for sentids, cocoid in zip(batch['sentids'], batch['cocoid']):
                for idx in sentids: 
                    temp_detected_entities, temp_caption, temp_clip_features, cocoid, temp_img_fearures = alldata_coco[idx]
                    # src_st = parse_entities([temp_detected_entities], stopwords, people_vocabs, objects_vocabs)[0]
                    src_st = compose_discrete_prompts(temp_detected_entities)
                    text_src.append(src_st)
                    text_tag.append(temp_caption)
                    img_features_list.append(temp_img_fearures)
                    rt_captions.append(torch.stack(list(map(cap_feat_coco.get, rt_caps_coco[str(cocoid)]))))      
                    cocoid_list.append(cocoid)
                    
            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "img_features_list": img_features_list,
                "cocoid_list": cocoid_list,
                "rt_captions": rt_captions
            }            

        # CFG preprocessing 
        if split == "train" and swap_cfg_coef:
            length = len(new_batch["text_src"])
            swaps = (np.random.rand(length) < swap_cfg_coef)
            new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]
            new_batch["text_trg_cond"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_trg"])]
            new_batch["rt_captions"] = [replace if swaps[i] else src for i, src in enumerate(new_batch["rt_captions"])]

        return new_batch
    
    if dataset_name in ["mscoco_ng"]:
        text_src = []
        text_tag = []
        img_features_list = []
        text_tag = []
        cocoid_list = []
        rt_captions = []

        for sentids in batch['sentids']:
            for idx in sentids: 
                temp_detected_entities, temp_caption, temp_clip_features, cocoid, temp_img_fearures = alldata_coco[idx]
                src_st = parse_entities([temp_detected_entities], stopwords, people_vocabs, objects_vocabs)[0]
                text_src.append(src_st)
                text_tag.append(temp_caption)
                img_features_list.append(temp_img_fearures.cpu())
                cocoid_list.append(cocoid)
                if split == 'train':
                    rt_captions.append(torch.stack(list(map(cap_feat_coco.get, rt_caps_coco[temp_caption.rstrip()]))))
                else:
                    rt_captions.append(torch.stack(list(map(cap_feat_coco.get, rt_caps_coco[str(cocoid)]))))      
        if split == 'train' :
            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "rt_captions": rt_captions
            }
        else:
            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "img_features_list": img_features_list,
                "cocoid_list": cocoid_list,
                "rt_captions": rt_captions
            }            

        # CFG preprocessing
        if split == "train" and swap_cfg_coef:
            length = len(new_batch["text_src"])
            swaps = (np.random.rand(length) < swap_cfg_coef)
            new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]

        return new_batch   
    
    if dataset_name in ["flickr30k"] or "flickr30k_ng" in dataset_name:
        text_src = []
        text_tag = []
        img_features_list = []
        text_tag = []
        cocoid_list = []
        rt_captions = []
        if split == 'train' :
            for sentids, cocoid in zip(batch['sentids'], batch['filename']):
                for idx in sentids: 
                    temp_detected_entities, temp_caption, temp_clip_features, cocoid, temp_img_fearures = alldata_flickr30k[int(idx)]
                    src_st = parse_entities([temp_detected_entities], stopwords, people_vocabs, objects_vocabs)[0]
                    text_src.append(src_st)
                    text_tag.append(temp_caption)
                    rt_captions.append(torch.stack(list(map(cap_feat_flickr30k.get, rt_caps_flickr30k[temp_caption.rstrip()]))))

            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "rt_captions": rt_captions
            }

        else:
            for sentids, cocoid in zip(batch['sentids'], batch['filename']):
                for idx in sentids: 
                    temp_detected_entities, temp_caption, temp_clip_features, _, temp_img_fearures = alldata_flickr30k[int(idx)]
                    src_st = compose_discrete_prompts(temp_detected_entities)
                    text_src.append(src_st)
                    text_tag.append(temp_caption)
                    img_features_list.append(temp_img_fearures)
                    rt_captions.append(torch.stack(list(map(cap_feat_flickr30k.get, rt_caps_flickr30k[cocoid]))))      
                    cocoid_list.append(cocoid)
                    
            new_batch = {
                "text_src": text_src,
                "text_trg": text_tag,
                "img_features_list": img_features_list,
                "cocoid_list": cocoid_list,
                "rt_captions": rt_captions
            }            

        # CFG preprocessing
        if split == "train" and swap_cfg_coef:
            length = len(new_batch["text_src"])
            swaps = (np.random.rand(length) < swap_cfg_coef)
            new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]
            new_batch["text_trg_cond"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_trg"])]
            new_batch["rt_captions"] = [replace if swaps[i] else src for i, src in enumerate(new_batch["rt_captions"])]

        return new_batch

    if dataset_name in ["nocaps"]:
        text_src = []
        text_tag = []
        img_features_list = []
        text_tag = []
        cocoid_list = []
        rt_captions = []
        for cocoid in zip(batch['image_id']):
            temp_detected_entities, temp_caption, temp_clip_features, _, temp_img_fearures = alldata_nocaps[str(cocoid[0])]
            src_st = compose_discrete_prompts(temp_detected_entities)
            text_src.append(src_st)
            text_tag.append(temp_caption)
            img_features_list.append(temp_img_fearures)
            rt_captions.append(torch.stack(list(map(cap_feat_coco.get, rt_caps_nocaps[str(cocoid[0])+'.jpg']))))      
            cocoid_list.append(cocoid[0])
        new_batch = {
            "text_src": text_src,
            "text_trg": text_tag,
            "img_features_list": img_features_list,
            "cocoid_list": cocoid_list,
            "rt_captions": rt_captions
        }            

        return new_batch
def batch_preprocessing(batch, dataset_name, split, swap_cfg_coef):
    if dataset_name == "qqp":
        if split == "train":
            p_swap = 0.5
        else:
            p_swap = 0.

        new_batch = {
            "text_src": [],
            "text_trg": [],
        }
        for src, trg in zip(batch["question1"], batch["question2"]):
            if np.random.rand() < p_swap:
                src, trg = trg, src
            new_batch["text_src"].append(src)
            new_batch["text_trg"].append(trg)
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]
        
    elif dataset_name == "xsum":
        new_batch = {
            "text_src": batch["document"],
            "text_trg": batch["summary"],
        }
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]

    elif dataset_name == "wiki_auto":
        new_batch = {
            "text_src": batch["source"],
            "text_trg": batch["target"],
            "references": batch["references"],
        }
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]

    elif dataset_name == "rocstories":
        new_batch = {
            "text_trg": batch["target"],
        }

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    # CFG preprocessing
    if split == "train" and swap_cfg_coef:
        length = len(new_batch["text_src"])
        swaps = (np.random.rand(length) < swap_cfg_coef)
        new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]
        
    return new_batch
