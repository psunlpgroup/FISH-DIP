# part of this code is modified from TANL - https://arxiv.org/abs/2101.05779

import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Trainer, T5Tokenizer
import re
from augment.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from augment.datasets_all import load_dataset
from augment.evaluate import evaluate, get_avg_results, print_results
from augment.utils import get_episode_indices
import numpy as np
from transformers import Adafactor

early_stop_count = 0
early_stop_loss_threshold = 1e-6
early_stop_patience = 20
chosen_params = []

# args
subsequent_param_percentage = 0.01
initial_param_percentage = 0.01
reevaluate_after_steps = 100


def calculate_gradients(model, data_loader, cuda_device, grad_type):

    losses = []
    gradients_dict = {}
    max_samples = 25
    min_samples = 3

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    # for calculating gradient, use a fraction of the available samples
    sample_percentage_used = 0.1
    num_samples = min(max(int(sample_percentage_used * len(data_loader)), min_samples), max_samples)


    tmp_dl = DataLoader(data_loader.dataset, batch_size=data_loader.batch_size * 2, collate_fn=data_loader.collate_fn,
                        pin_memory=data_loader.pin_memory)

    with torch.no_grad():
        for idx, inputs in enumerate(tmp_dl):
            inputs.pop("idx", None)

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(cuda_device)

            return_dicts = model(**inputs)
            lm_logits = return_dicts['logits']
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            _loss = (loss_fct(lm_logits.view(-1, lm_logits.size(-1)), inputs['labels'].view(-1))).view(lm_logits.size(0), -1)
            loss = torch.mean(_loss, axis=1)
            loss = loss.detach().cpu().numpy()
            losses = np.append(losses, loss, axis=0)

    idxes = np.argpartition(losses, -min(num_samples, len(losses) - 1))[-min(num_samples, len(losses) - 1):] # largest num sample indices

    # do it sample by sample.
    # TODO much faster implementation -> batchify the samples

    subset = torch.utils.data.Subset(data_loader.dataset, idxes)
    tmp_dl = DataLoader(subset, batch_size=1, collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory)

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    for idx, inputs in enumerate(tmp_dl):

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)
        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data

        model.zero_grad()

    return gradients_dict


def create_mask_gradient(model, helperTrainer, keep_ratio, sample_type, grad_type, include_match_str = ""):
    global chosen_params
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = helperTrainer.get_train_dataloader()

    data_loader_seq = DataLoader(
        dataset=data_loader.dataset,
        batch_size=data_loader.batch_size,
        sampler=SequentialSampler(data_loader.dataset),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
    )

    if sample_type == "label":
        importance_method = calculate_gradients
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader_seq, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []
    classifier_size = 0
    extra_inclusion_size = 0
    all_params_size = 0
    classifier_mask_dict = {}
    inclusion_mask_dict = {}


    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)

        elif re.fullmatch(include_match_str, k):
            extra_inclusion_size += torch.prod(torch.tensor(v.shape)).item()
            inclusion_mask_dict[k] = torch.ones_like(v).to(original_device)

        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size - extra_inclusion_size * 0

    assert keep_num > 0
    top_pos = torch.topk(tensors, keep_num)[1]
    masks = torch.zeros_like(tensors, device=cuda_device)
    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)
    mask_dict.update(inclusion_mask_dict)

    model.to(original_device)

    return mask_dict

total_steps = 0
total_steps_debug = 2000

class SparseUpdateTrainer(Trainer):
    def __init__(self, *args, mask, tempTrainer, params_to_keep, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.tempTrainer  = tempTrainer
        self.params_to_keep = params_to_keep

    def training_step(self, *args, **kwargs):
        global total_steps, early_stop_loss_threshold, early_stop_count, early_stop_patience
        total_steps += 1
        if total_steps % total_steps_debug == 0:

            print("Steps done: " + str(total_steps_debug))

        if total_steps % reevaluate_after_steps == 0:
            self.mask = create_mask_gradient(
                self.model,
                self.tempTrainer,
                subsequent_param_percentage,
                'label',
                'square',
                self.params_to_keep
            )

        loss = super().training_step(*args, **kwargs)

        # Early stopping is optional Might be useful.
        if loss < early_stop_loss_threshold:
            early_stop_count += 1

            if early_stop_count >= early_stop_patience:
                self.control.should_training_stop = True
                logging.info("-------Training early stopped due to hitting early stop patience limit---------")
        else:
            early_stop_count = 0


        # mask out the gradients
        for name, params in self.model.named_parameters():
            device = params.device
            self.mask[name] = self.mask[name].to(device)
            params.grad.data.copy_(params.grad.data * self.mask[name].data)

        return loss


def main():
    assert torch.cuda.is_available(), 'CUDA not available'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--algorithm', type=str, help='choose either label or expect', default='label')
    parser.add_argument('--percentage', type=float, help='percentage of parameters to keep trainable in each X step', default=0.01)
    parser.add_argument('--subsequent_param_percentage', type=float, help='percentage of parameters to keep trainable in each X step', default=0.01)
    parser.add_argument('--method', type=str, help='absolute or square', default='square')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')

    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')
    parser.add_argument('--reevaluate_after_steps', type=int, default=100, help='How many steps after you want to evaluate')

    args, remaining_args = parser.parse_known_args()

    global reevaluate_after_steps
    reevaluate_after_steps = args.reevaluate_after_steps

    # read config file
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    train_subset, num_train_epochs, model_name_or_path = None, None, None

    for k in range(len(remaining_args)):
        if remaining_args[k] == '--train_subset':
            train_subset = (remaining_args[k + 1])

        if remaining_args[k] == '--num_train_epochs':
            num_train_epochs = (remaining_args[k + 1])

        if remaining_args[k] == '--model_name_or_path':
            model_name_or_path = remaining_args[k+1]

    global subsequent_param_percentage
    subsequent_param_percentage = args.subsequent_param_percentage

    name = model_name_or_path + '_' + args.job + '_percentage-' + str(args.percentage) + '_refresh-perc-' + str(subsequent_param_percentage) + '_subset-' + str(train_subset) + '_epochs-' + str(num_train_epochs) + 'reeval_steps-' + str(reevaluate_after_steps)

    # set defaults for other arguments
    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 12,
        'logging_steps': 5,  # do not log by default
        'save_steps': 0,  # do not save checkpoints by default
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            # interpret True/False as boolean
            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':
            # interpret as None
            defaults[key] = None

    if args.eval:
        # run evaluation only
        defaults['do_train'] = False

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length

    if data_args.chunk_size_eval is None:
        # defaults to chunk_size
        data_args.chunk_size_eval = data_args.chunk_size

    if data_args.chunk_overlap_eval is None:
        # defaults to chunk overlap
        data_args.chunk_overlap_eval = data_args.chunk_overlap

    # construct name for the output directory
    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}',
        f'-{name}',
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-ep{round(training_args.num_train_epochs)}'
        f'-len{data_args.max_seq_length}'
    )

    if data_args.max_output_seq_length != data_args.max_seq_length:
        output_dir += f'-{data_args.max_output_seq_length}'

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.chunk_size != 128:
        output_dir += f'-chunk{data_args.chunk_size}'
    if data_args.chunk_overlap != 64:
        output_dir += f'-overlap{data_args.chunk_overlap}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    if data_args.train_subset < 1:
        output_dir += f'-size{data_args.train_subset:.2f}'

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    # setup logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'logs.log'),
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # construct file name for the evaluation results
    evaluation_output_filename = f'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'

    # create model config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # create tokenizer
    model_args.tokenizer_name = 't5-large'

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,force_download=True
    )

    # get list of dataset names
    dataset_names = data_args.datasets.split(',')

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)

    # episode loop
    # (note that the episode index is used as the random seed, so that each episode is reproducible)
    evaluation_results = defaultdict(list)
    for ep_idx in episode_indices:
        print()
        logging.info(f'Episode {ep_idx} ({len(episode_indices)} episodes total)')
        episode_output_dir = os.path.join(output_dir, f'episode{ep_idx}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        logging.info(f'Output directory: {episode_output_dir}')

        training_args.output_dir = episode_output_dir  # checkpoints are saved in episode-specific directory

        # load pretrained model
        model = None
        str_to_include_params = ""

        if training_args.zero_shot or training_args.do_train:
            logging.info(f"Using model {model_args.model_name_or_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )

        # fine-tune the model
        if training_args.do_train:
            # load train dataset
            datasets = []
            for dataset_name in dataset_names:
                logging.info(f'Process dataset {dataset_name} (train)')
                dataset = load_dataset(
                    dataset_name, data_args, split=data_args.train_split,
                    max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                    tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
                )
                datasets.append(dataset)

            train_dataset = torch.utils.data.ConcatDataset(datasets) if training_args.do_train else None

            optimizer = Adafactor(
                model.parameters(),
                lr=1e-2,
                weight_decay=0,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
            )


            tempTrainer = Trainer(model=model,
                args=training_args,
                train_dataset=train_dataset)

            mask = create_mask_gradient(
                model,
                tempTrainer,
                args.percentage,
                args.algorithm,
                args.method,
                str_to_include_params,
            )

            trainer = SparseUpdateTrainer(
                model=model,
                args=training_args,
                mask=mask,
                tempTrainer = tempTrainer,
                train_dataset=train_dataset,
                params_to_keep = str_to_include_params,
                optimizers=(optimizer, None)

            )

            # start trainer
            logging.info('Start training')
            trainer.train()

            trainer.save_model(episode_output_dir)


        # run evaluation
        if training_args.local_rank in [-1, 0] and (training_args.do_eval or training_args.do_predict):
            # should we evaluate on dev, test, or both?
            evaluation_splits = []
            if training_args.do_eval:
                evaluation_splits.append('dev')
            if training_args.do_predict:
                evaluation_splits.append('test')

            # should we evaluate on the final model and/or on all intermediate checkpoints?
            evaluation_dirs = []
            evaluation_dirs += ['']

            # datasets to evaluate on
            if data_args.eval_datasets is None:
                eval_dataset_names = dataset_names
            else:
                eval_dataset_names = data_args.eval_datasets.split(',')

            # evaluate all possible combinations of dev/test, model, and datasets
            for comb in itertools.product(evaluation_splits, evaluation_dirs, eval_dataset_names):
                split, evaluation_dir, dataset_name = comb
                model_dir = os.path.join(episode_output_dir, evaluation_dir)

                if model is None:
                    # we need to load the model
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_dir,
                        config=config,
                    )

                if len(evaluation_dir) > 0:
                    logging.info(f'Evaluate {evaluation_dir} on {dataset_name} {split}')
                else:
                    logging.info(f'Evaluate on {dataset_name} {split}')

                res = evaluate(
                    model=model, dataset_name=dataset_name, data_args=data_args, tokenizer=tokenizer, split=split,
                    seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu
                )
                # store results
                evaluation_results[comb].append(res)

                # print results
                if args.verbose_results:
                    print_results(res)

                # save results to file
                with open(
                        os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}.json'), 'w'
                ) as f:
                    json.dump(res, f, indent=0)

    # print average results and save them to file
    for comb, results in evaluation_results.items():
        split, evaluation_dir, dataset_name = comb

        print()
        logging.info(
            f'Average of {split} results over {len(results)} episodes ({dataset_name} {evaluation_dir}):'
        )
        res = get_avg_results(results)

        # print average results
        print_results(res)
        for key, value in res.items():
            s = key
            print(s)
            if isinstance(value, (list, tuple)):
                mean, std = value
                print({s + '_mean': mean})
                print({s + '_std': std})

        # save average results to file
        filename = evaluation_output_filename + f'-{dataset_name}-{split}'
        if len(evaluation_dir) > 0:
            filename += '-'
        filename += f'{evaluation_dir}.json'

        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(res, f, indent=0)

    print()
    logging.info(f'Model weights and intermediate checkpoints saved in {output_dir}')


if __name__ == "__main__":
    main()
