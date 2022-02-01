import json
import pickle
from typing import Dict
import argparse
from argparse import Namespace
import glob
import random
import numpy as np
import itertools
import logging
logger = logging.getLogger(__name__)

# pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# pytorch lightning packages
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# huggingface transformers packages
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import (
	Adafactor,
	get_cosine_schedule_with_warmup,
	get_cosine_with_hard_restarts_schedule_with_warmup,
	get_linear_schedule_with_warmup,
	get_polynomial_decay_schedule_with_warmup,
)

# allennlp dataloading packages
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.token import Token

# Globe constants
training_size = 684100
# validation_size = 145375

# log_every_n_steps how frequently pytorch lightning logs.
# By default, Lightning logs every 50 rows, or 50 training steps.
log_every_n_steps = 1

arg_to_scheduler = {
	"linear": get_linear_schedule_with_warmup,
	"cosine": get_cosine_schedule_with_warmup,
	"cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
	"polynomial": get_polynomial_decay_schedule_with_warmup,
	# '': get_constant_schedule,             # not supported for now
	# '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class DataReaderFromPickled(DatasetReader):
	"""
	This is copied from https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/data.py#L61:L109 without any change
	"""
	def __init__(self,
				 lazy: bool = False,
				 word_splitter: WordSplitter = None,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 max_sequence_length: int = 256,
				 concat_title_abstract: bool = None
				 ) -> None:
		"""
		Dataset reader that uses pickled preprocessed instances
		Consumes the output resulting from data_utils/create_training_files.py

		the additional arguments are not used here and are for compatibility with
		the other data reader at prediction time
		"""
		self.max_sequence_length = max_sequence_length
		self.token_indexers = token_indexers
		self._concat_title_abstract = concat_title_abstract
		super().__init__(lazy)

	def _read(self, file_path: str):
		"""
		Args:
			file_path: path to the pickled instances
		"""
		with open(file_path, 'rb') as f_in:
			unpickler = pickle.Unpickler(f_in)
			while True:
				try:
					instance = unpickler.load()
					# compatibility with old models:
					# for field in instance.fields:
					#     if hasattr(instance.fields[field], '_token_indexers') and 'bert' in instance.fields[field]._token_indexers:
					#         if not hasattr(instance.fields['source_title']._token_indexers['bert'], '_truncate_long_sequences'):
					#             instance.fields[field]._token_indexers['bert']._truncate_long_sequences = True
					#             instance.fields[field]._token_indexers['bert']._token_min_padding_length = 0
					if self.max_sequence_length:
						for paper_type in ['source', 'pos', 'neg']:
							if self._concat_title_abstract:
								tokens = []
								title_field = instance.fields.get(f'{paper_type}_title')
								abst_field = instance.fields.get(f'{paper_type}_abstract')
								if title_field:
									tokens.extend(title_field.tokens)
								if tokens:
									tokens.extend([Token('[SEP]')])
								if abst_field:
									tokens.extend(abst_field.tokens)
								if title_field:
									title_field.tokens = tokens
									instance.fields[f'{paper_type}_title'] = title_field
								elif abst_field:
									abst_field.tokens = tokens
									instance.fields[f'{paper_type}_title'] = abst_field
								else:
									yield None
								# title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
								# pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
								# neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
								# query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []
							for field_type in ['title', 'abstract', 'authors', 'author_positions']:
								field = paper_type + '_' + field_type
								if instance.fields.get(field):
									instance.fields[field].tokens = instance.fields[field].tokens[
																	:self.max_sequence_length]
								if field_type == 'abstract' and self._concat_title_abstract:
									instance.fields.pop(field, None)
					yield instance
				except EOFError:
					break


class IterableDataSetMultiWorker(IterableDataset):
	def __init__(self, file_path, tokenizer, size, block_size=100):
		self.datareaderfp = DataReaderFromPickled(max_sequence_length=512)
		self.data_instances = self.datareaderfp._read(file_path)
		self.tokenizer = tokenizer
		self.size = size
		self.block_size = block_size

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			iter_end = self.size
			for data_instance in itertools.islice(self.data_instances, iter_end):
				data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
				yield data_input

		else:
			# when num_worker is greater than 1. we implement multiple process data loading.
			iter_end = self.size
			worker_id = worker_info.id
			num_workers = worker_info.num_workers
			i = 0
			for data_instance in itertools.islice(self.data_instances, iter_end):
				if int(i / self.block_size) % num_workers != worker_id:
					i = i + 1
					pass
				else:
					i = i + 1
					data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
					yield data_input

	def ai2_to_transformers(self, data_instance, tokenizer):
		"""
		Args:
			data_instance: ai2 data instance
			tokenizer: huggingface transformers tokenizer
		"""
		source_tokens = data_instance["source_title"].tokens
		source_title = tokenizer(' '.join([str(token) for token in source_tokens]),
								 truncation=True, padding="max_length", return_tensors="pt", return_token_type_ids=True,
								 max_length=512)

		source_input = {'input_ids': source_title['input_ids'][0],
						'token_type_ids': source_title['token_type_ids'][0],
						'attention_mask': source_title['attention_mask'][0]}

		pos_tokens = data_instance["pos_title"].tokens
		pos_title = tokenizer(' '.join([str(token) for token in pos_tokens]),
							  truncation=True, padding="max_length", return_token_type_ids=True, return_tensors="pt", max_length=512)

		pos_input = {'input_ids': pos_title['input_ids'][0],
					 'token_type_ids': pos_title['token_type_ids'][0],
					 'attention_mask': pos_title['attention_mask'][0]}

		neg_tokens = data_instance["neg_title"].tokens
		neg_title = tokenizer(' '.join([str(token) for token in neg_tokens]),
							  truncation=True, padding="max_length", return_token_type_ids=True, return_tensors="pt", max_length=512)

		neg_input = {'input_ids': neg_title['input_ids'][0],
					 'token_type_ids': neg_title['token_type_ids'][0],
					 'attention_mask': neg_title['attention_mask'][0]}

		return source_input, pos_input, neg_input


class IterableDataSetMultiWorkerTestStep(IterableDataset):
	def __init__(self, file_path, tokenizer, size, block_size=100):
		self.datareaderfp = DataReaderFromPickled(max_sequence_length=512)
		self.data_instances = self.datareaderfp._read(file_path)
		self.tokenizer = tokenizer
		self.size = size
		self.block_size = block_size

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			iter_end = self.size
			for data_instance in itertools.islice(self.data_instances, iter_end):
				data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
				yield data_input

		else:
			# when num_worker is greater than 1. we implement multiple process data loading.
			iter_end = self.size
			worker_id = worker_info.id
			num_workers = worker_info.num_workers
			i = 0
			for data_instance in itertools.islice(self.data_instances, iter_end):
				if int(i / self.block_size) % num_workers != worker_id:
					i = i + 1
					pass
				else:
					i = i + 1
					data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
					yield data_input

	def ai2_to_transformers(self, data_instance, tokenizer):
		"""
		Args:
			data_instance: ai2 data instance
			tokenizer: huggingface transformers tokenizer
		"""
		source_tokens = data_instance["source_title"].tokens
		source_title = tokenizer(' '.join([str(token) for token in source_tokens]),
								 truncation=True, padding="max_length", return_tensors="pt",
								 max_length=512)
		source_input = {'input_ids': source_title['input_ids'][0],
						'token_type_ids': source_title['token_type_ids'][0],
						'attention_mask': source_title['attention_mask'][0]}

		source_paper_id = data_instance['source_paper_id'].metadata

		return source_input, source_paper_id


if __name__ == '__main__':
#	tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
	train_dataset = DataReaderFromPickled() #IterableDataSetMultiWorker(file_path='data/train.pkl', tokenizer=tokenizer, size=512)
	index = 0
	with open('data/val.tsv', 'w') as train_file:
		for train_instance in train_dataset._read('data/val.pkl'):
			train_instance_source = train_instance['source_title'].tokens
			train_instance_source = ' '.join([str(token) for token in train_instance_source])
			train_instance_pos = train_instance['pos_title'].tokens
			train_instance_neg = train_instance['neg_title'].tokens
			train_instance_neg = " ".join([str(token) for token in train_instance_neg])
			train_instance_pos = " ".join([str(token) for token in train_instance_pos])
			index +=1
			print(index)
			train_file.write(train_instance_source + '\t' + train_instance_pos + '\t' + train_instance_neg + '\n')
		#print(train_instance_source, train_instance_neg, train_instance_pos)

