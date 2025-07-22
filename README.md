# ZDC-AAAI
## ZDC: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings

![Results](pic\fig1.png)

Abstract: *Zero‑shot captioning can generate descriptions for visual input without requiring human-annotated data. While autoregressive models with beam search achieve strong accuracy, they are prone to mode collapse because beam search tends to favor high-probability sequences, leading to generic captions with little variability. To address this issue, we propose a novel diffusion-based framework for zero-shot diverse captioning that separates the generation of diverse text encodings from an anti-collapse decoding process. First, we introduce a prompt-conditioned diffusion prior, which leverages the inherent stochasticity of the diffusion process to generate highly diverse text encodings. Second, we develop a multi-level negative training loss that steers the decoder away from trivial modes. Extensive experiments demonstrate that our diffusion-based framework, combined with negative teacher training, strikes a superior balance between captioning accuracy and output diversity. Our code is available at https://anonymous.4open.science/r/ZDC-AAAI-A9BB.*

![Framework](pic\framework.png)

## Requirements

* Python libraries: See [requirements.txt](./requirements.txt) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda create --name zdc python=3.9`
  - `conda activate zdc`
  - `conda install pip`
  - `pip install -r requirements.txt`
  - `python -m spacy download en`

## Dataset loading

You can download any dataset used in the article with the following command:

```
python -m data.load --dataset_name=dataset_name
```

where `'dataset_name'` is one of the following:
 - `'mscoco'`
 - `'flickr30k'`

By default your datasets will be saved in the folder `'datasets'` in your working directory. If you want to specify another path, use the argument `'dataset_path'`
## Statistics computation

For proper diffusion model training you need to compute dataset statistics for any particular dataset and encoder with the following command:

```
python -m data.make_statistics --dataset_name=dataset_name --encoder_name=encoder_name
```

where `'encoder_name'` is one the following:
 - `'t5-base'`

Notice that default TEncDM encoder is `'t5-base'`, but you can experiment with it


## Decoder training
Just as the statistics, you need to train a decoder for every pair of dataset and encoder. You can do this with the command:

```
python -m model.train_decoder --dataset_name=dataset_name --encoder_name=encoder_name
```

## Diffusion training

Finally, after training the decoder and computing dataset statistics, you can start training yout diffusion model. To train basic TEncdDM setup, run

```
torchrun --nproc_per_node=n train_diffusion.py --dataset_name=dataset_name --encoder_name=encoder_name
```

You will train the best TEncDM network with this script. But you can also test other hyperparameters, such as:

- `--scheduler` - noise scheduler to use,`default='sd'`. Can be one of the following: `'sqrt'`, `'cosine'`, `'sd'`
- `--coef_d` - coefficient for tangent scheduler, `default=9`
- `--swap_cfg_coef` - classifier free guidance coefficient, `default=0.0`
- `--project_name` - wandb project name, `default='tencdm'`
- `--N` - sample steps, `default='50'`
- `--test_cfg` - classifier free guidance coefficient (inference), `default='0.5'`
