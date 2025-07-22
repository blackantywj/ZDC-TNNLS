from typing import Tuple, List, Union, Optional, Dict, Any
import torch
from collections import UserDict
import random

def tokenize(batch_texts: List[str], tokenizer, max_length):
    X = tokenizer(
        batch_texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    return X

class BatchEncoding(UserDict):
    def __init__(self, data: Optional[Dict[str, Any]] = None, return_tp="pt"):
        super().__init__(data)
        self.return_tp = return_tp

    def __getitem__(self, item: str) -> Union[str, torch.Tensor]:
        if isinstance(item, str):
            value = self.data[item]
            if item.startswith("text"):
                return value
            else:
                if self.return_tp == "pt":
                    return torch.Tensor(value)
                else:
                    return value
        elif isinstance(item, slice):
            return BatchEncoding({key: self.data[key][item] for key in self.data.keys()})
        else:
            raise KeyError(
                "Invalid key. Only two types of key are available: "
                "(1) string, and (2) slices for data subsetting."
            )

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        for k, v in self.data.items():
            if not k.startswith("text"):
                self.data[k] = v.to(device=device)
        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

def parse_entities(
    detected_entities: Tuple[str],      # [[man, dog, park, ...], len = batch size
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[torch.Tensor]:
    # List[(n_seq1, ), (n_seq2, ), ...]

    discrete_tokens = []
    for idx in range(len(detected_entities)):
        # entities processing
        process_entities = entities_process(detected_entities[idx], stopwords, people_vocabs, objects_vocabs)
        process_entities = list(set(process_entities)) # list

        # tokenizing
        discrete_tokens.append(compose_discrete_prompts(process_entities))

    return discrete_tokens

def entities_process(
    detected_entities: List[str],  # [man, dog, park]
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[str]:
    process_entities = []
    for i in range(len(detected_entities)):
        if i >= 10: # There is no entity detected
            break
        detected_entity = detected_entities[i]                # processing the i-th entity
        if detected_entity in people_vocabs:                  # transforming all person concept (man, woman, kid, ...) to the same word 'person'
            detected_entity = 'person'
        elif len(detected_entity) > 1 and detected_entity not in stopwords and detected_entity in objects_vocabs: # only processing entities in visual genome
            pass
        else: # processing the next entities
            continue

        if True:
            random_prob = random.random()
            if random_prob < 0.4:         # mask
                pass
            else:                                              # remain
                process_entities.append(detected_entity)

    return process_entities

def compose_discrete_prompts(
    process_entities: List[str],
) -> torch.Tensor:

    prompt_head = 'There are'
    prompt_tail = ' in image.'

    if len(process_entities) == 0: # without entities
        discrete_prompt =  prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities: # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','     # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]        # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail # 'There are person, dog, park in image.'

    # entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))   # (discrete_prompt_length, ) 

    return discrete_prompt

def load_stopwords() -> List[str]:
    # Return: stopwords and punctuations

    stopwords = {'per', '’ll', 'could', 'fifteen', 'been', "isn't", 'whoever', 'any', 'whole', 'front', "won't", 'upon', 'there', 's', 'am', 'via', 'the', 'as', "haven't", 'on', 'km', 'further', 'their', 'quite', 'have', 'twenty', 'during', 'full', 'it', 'thin', 'so', 'what', 'an', 't', 'less', 'if', 'sixty', 'everyone', 'us', 'were', 'side', 'she', 'cannot', 'thereby', '‘ve', 'amount', 'n’t', 'be', 'nine', 'isn', 'wouldn', 'by', 'along', "'ll", 'themselves', 'forty', 'everywhere', "'d", 'thru', 'sometimes', 'hasnt', 'seeming', 'own', 'that', "'ve", 'least', 'with', 'inc', 'really', 'afterwards', 'due', 'for', 'sometime', 'last', 'find', 'therein', 'all', 'thick', 'detail', 'few', 'hundred', 'some', 'even', 'off', '’m', 'ain', '’re', 'hence', 'etc', 'into', 'rather', 'where', 'm', 'its', 'onto', '’s', 'get', 'other', 'moreover', 'noone', 'being', 'must', 'bill', "wasn't", 'system', 'neither', "you'll", 'third', 'whereby', 'nobody', 'among', 'throughout', 'except', 'beforehand', "didn't", 'was', 'without', 'whose', 'hasn', '‘d', 'or', 'theirs', 'various', 'name', 'twelve', 'myself', 'former', 'though', 'we', 'ours', 'many', 'sincere', 'regarding', 'had', 'before', 'mustn', 'either', 'doing', 'why', 'fill', 'eight', 'won', 'anything', 'hereupon', 'this', 'amoungst', '‘s', 'of', 'yourselves', 'beside', 'within', 'ourselves', '‘re', 'about', 'elsewhere', 'latter', 'through', 'll', 'i', 'wasn', 'anywhere', 'weren', 'just', 'itself', "you're", 'wherein', 'four', 'keep', 'whether', 'nothing', 'found', 'back', 'needn', "aren't", 'has', 'one', 'wherever', 'serious', 'everything', 'hadn', 'first', 'anyway', 'co', 'still', 'five', 'becomes', "don't", 'formerly', 'ever', 'part', 'nowhere', 'made', 'himself',  "couldn't", 'none', 'others', 'now', 'doesn', 'at', 'another', 'does', 'kg', 'see', 'often', 'them', 'shan', 'fifty', 'ltd', 'namely', 'they', 'somewhere', 'haven', 'take', 'latterly', 'well', 'whatever', 'nor', 'whereafter', 'might', 'only', 'de', 'our', 'hers', "mustn't", 'aren', 'you', 'his', "wouldn't", 'please', 'empty', 'but', 'mightn', 'then', 'should', 'and', 'each', 'such', 'a', 'yet', 'y', 'enough', 'someone', 'would', 'since', 'however', 'make', 'alone', 'anyone', 'amongst', 'these', 'whereupon', 'fire', "hasn't", 'shouldn', 'didn', 'do', 'me', 'becoming', 'after', 'several', 'seem', 'her', 'three', 'out', 'ten', 'whence', 'eg', 'couldn', 'un', 'did', "she's", 'whither', 'toward', 'once', "should've", 'call', "weren't", 'again', 'more', 'show', 'seems', "needn't", 'thereupon', 'used', 'most', 'hereby', 'put', 'ie', 've', 'my', 'your', 'thence', 'already', 'always', 'having', 'much', 'move', 'eleven', "'re", 'here', 'yours', 'con', 'done', 'up', 'over', 'yourself', "it's", 'o', 'six', 'can', 'how', "hadn't", 'anyhow', 'below', 'also', 'say', 'together', 'down', 'using', 'while', 'almost', 'cry', "you've", '’ve', 'two', 'towards', 'meanwhile', 'perhaps', 'when', 'ma', "shouldn't", 'both', 'hereafter', 'he', 'describe', 'ca', 'which', 'every', 'between', 'give', 'go', 'very', '’d', 'nevertheless', 'is', 'n‘t', 'therefore', '‘ll', 'unless', 'next', 'who', 'became', 'mill', 'him', 'don', 'same', "'s", 'seemed', 'mostly', 'will', 're', "you'd", 'no', 'in', 'too', "mightn't", 'besides', 'are', 'because', 'couldnt', 'd', 'against', "doesn't", 'cant', 'whenever', 'somehow', 'thereafter', 'although', 'beyond', 'from', 'whereas', 'thus', 'than', "shan't", 'to', 'top', 'until', 'those', 'whom', 'bottom', 'else', 'herein', 'something', '‘m', 'may', 'not', "that'll", "'m", 'indeed', 'never', 'herself', 'interest', "n't", 'become', 'mine', 'otherwise'}
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    other_words = {'photo', 'image', 'picture', 'pic', 'side', 'part', 'background'}
    stopwords_and_punctuations = stopwords.union(punctuations)
    stopwords_and_punctuations = stopwords_and_punctuations.union(other_words)
    stopwords_and_punctuations = [stopword.lower() for stopword in stopwords_and_punctuations]
    stopwords_and_punctuations.sort()

    return stopwords_and_punctuations