import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm
import logging
import argparse
# import statement for measuring time
import time
import re

REVISION_PROMPT_FILE_NAME = 'revision_prompts.txt'
logging.basicConfig(level=logging.INFO)

PROMPTS_TXT = 'prompts.txt'
PREDICTIONS_TXT = 'predictions.txt'
# map from language code to language name
LANGUAGES = {
    'hin_Deva': 'Hindi',
    'mar_Deva': 'Marathi',
    'ben_Beng': 'Bengali',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'kan_Knda': 'Kannada',
    'mal_Mlym': 'Malayalam',
    'guj_Gujr': 'Gujarati',
    'ori_Orya': 'Odia',
    'asm_Asan': 'Assamese',
    'nep_Nepa': 'Nepali',
    'pan_Guru': 'Punjabi',
    'san_Sinh': 'Sinhala',
    'snd_Deva': 'Sindhi',
    'ind_Latn': 'Indonesian',
    'zsm_Latn': 'Malay',
    'por_Latn': 'Portuguese',
    'spa_Latn': 'Spanish',
    'rus_Cyrl': 'Russian',
    'ukr_Cyrl': 'Ukrainian'
}


def load_samples(path):
    with open(path, 'r') as f:
        content = f.read()
        samples = content.splitlines()
    return samples


def get_samples(src_lang_code, dst_lang_code, mode, input_file, output_file):
    if mode == 'dev':
        src_test_path = 'data/{}.dev'.format(src_lang_code)
        dst_test_path = 'data/{}.dev'.format(dst_lang_code)
        src_test_samples = load_samples(src_test_path)[5:]
        dst_test_samples = load_samples(dst_test_path)[5:]
    elif mode == 'test':
        src_test_path = 'data/{}.devtest'.format(src_lang_code)
        dst_test_path = 'data/{}.devtest'.format(dst_lang_code)
        src_test_samples = load_samples(src_test_path)
        dst_test_samples = load_samples(dst_test_path)
    else:
        src_test_path = 'data/{}'.format(input_file)
        dst_test_path = 'data/{}'.format(output_file)
        src_test_samples = load_samples(src_test_path)
        dst_test_samples = load_samples(dst_test_path)
        # only use the first 100 samples
        src_test_samples = src_test_samples[:10]
        dst_test_samples = dst_test_samples[:10]

    return src_test_samples, dst_test_samples


def process(tokenizer, model, mode, source_lang, target_lang, input_file, output_file, prompt_template, revision_template, batch_size=5, chunk_size=-1, root_output_directory=None):
    src_test_samples, dst_test_samples = get_samples(source_lang, target_lang, mode, input_file, output_file)
    # empty the prompts, predictions, and revision prompts files
    # create empty files for prompts with the part number
    prompt_file = os.path.join(root_output_directory, PROMPTS_TXT)
    prediction_file = os.path.join(root_output_directory, PREDICTIONS_TXT)
    revision_prompt_file = os.path.join(root_output_directory, REVISION_PROMPT_FILE_NAME)
    source_lang_ = LANGUAGES[source_lang]
    target_lang_ = LANGUAGES[target_lang]
    # Pre-compiled regular expression for optimized string extraction
    extraction_pattern = re.compile(r'<extra_id_0>(.*?)(' + '|'.join(
        [re.escape(source_lang_), re.escape(target_lang_), 'Translate', '<extra_id_1>']) + ')')
    fallback_pattern = re.compile(r'<extra_id_0>(.*?)</s>')
    open(prompt_file, 'w').close()
    open(prediction_file, 'w').close()
    open(revision_prompt_file, 'w').close()
    for src_test_sample_index in range(0, len(src_test_samples), batch_size):
        dataset_obj = MyDataset()
        # get the prompts for the next 5 samples
        src_test_samples_subset = src_test_samples[src_test_sample_index:src_test_sample_index + batch_size]
        dst_test_samples_subset = dst_test_samples[src_test_sample_index:src_test_sample_index + batch_size]
        # construct the prompts for the next 5 samples
        chunk_count_list = []
        for src_test_sample in src_test_samples_subset:
            prompts = construct_prompt(src_test_sample, source_lang=source_lang_, target_lang=target_lang_, prompt_template=prompt_template, chunk_size=chunk_size)
            dataset_obj.extendprompts(prompts)
            chunk_count_list.append(len(prompts))
        max_chunk_count = max(chunk_count_list) + 1
        # log the prompts which is a list of prompts
        logging.debug("dataset_obj.prompts %s", dataset_obj.prompts)
        # iterate through the indices in sent_index
        # the sent_index is an index of length of longest test sample
        # for each index, get the nth token from the src_test_samples
        # get the length of the longest test sample
        len_phrases_list = [len(split_phrases(sample, chunk_size=chunk_size)) for sample in src_test_samples_subset]
        max_length = max(len_phrases_list)
        logging.info("max length: {}".format(max_length))
        rev_dataset_obj = MyDataset()
        # iterate till the max_length
        prev_revised_predictions = None
        # get the predictions for the prompts; divide the dataset_obj.prompts into sets of 10 prompts and then run the model on each set
        dataset_obj_prompts = dataset_obj.prompts
        predictions = []
        batch_size_first_stage = 8
        for prompt_index in range(0, len(dataset_obj_prompts), batch_size_first_stage):
            dataset_obj_each = MyDataset()
            dataset_obj_each.setprompts(dataset_obj_prompts[prompt_index:prompt_index + batch_size_first_stage])
            predictions_each = predict_output(tokenizer, model, dataset_obj_each, extraction_pattern=extraction_pattern,
                                         fallback_pattern=fallback_pattern)
            predictions.extend(predictions_each)

        # Create predictions_sublists by padding with empty elements
        # Initialize variables to keep track of the current index and predictions_sublists
        chunk_index = 0
        predictions_sublists = []

        # Iterate through the lengths and divide biglist accordingly
        for chunk_count in chunk_count_list:
            sublist = predictions[chunk_index:chunk_index + chunk_count]
            # Pad the sublist with empty elements if needed
            while len(sublist) < max_chunk_count:
                sublist.append('')
            predictions_sublists.append(sublist)
            chunk_index += chunk_count
        for sent_index in range(1, max_length + 1):
            predictions_at_index = [sublist[sent_index] for sublist in predictions_sublists]
            # method to write prompt to a prompt file
            logging.info(f"sent_index: {sent_index}")
            logging.info(f"predictions: {predictions_at_index[0]}")
            # print the nth token from the src_test_samples
            # handle the case where the nth token is not present
            if sent_index < len_phrases_list[0]:
                logging.info(f"src_test_samples: {split_phrases(src_test_samples_subset[0], chunk_size=chunk_size)[sent_index]}")

            # iterate through the examples in the src_test_samples_subset and construct the prompts
            if sent_index == 1:
                for src_test_sample, prediction in zip(src_test_samples_subset, predictions_at_index):
                    revision_prompt = construct_revision_prompt(src_test_sample, prediction, previous_prompt=None, index=sent_index - 1, prev_revised_prediction=None, source_lang=source_lang_, target_lang=target_lang_, revision_template=revision_template, chunk_size=chunk_size)
                    rev_dataset_obj.addprompt(revision_prompt)
            else:
                revision_prompts = []
                for src_test_sample, prediction, previous_prompt, prev_revised_prediction, prediction_list in zip(src_test_samples_subset, predictions_at_index, rev_dataset_obj.prompts, prev_revised_predictions, rev_dataset_obj.predictions_list):
                    revision_prompt = construct_revision_prompt(src_test_sample, prediction, previous_prompt=previous_prompt, index=sent_index - 1, prev_revised_prediction=prev_revised_prediction, source_lang=source_lang_, target_lang=target_lang_, revision_template=revision_template, prediction_list=prediction_list, chunk_size=chunk_size)
                    revision_prompts.append(revision_prompt)
                rev_dataset_obj.setprompts(revision_prompts)
            revised_predictions = predict_output(tokenizer, model, rev_dataset_obj, revised=True, len_phrases_list=len_phrases_list, sent_index=sent_index, extraction_pattern=extraction_pattern, fallback_pattern=fallback_pattern)
            logging.info(f"revised predictions: {revised_predictions[0]}")
            prev_revised_predictions = revised_predictions
            rev_dataset_obj.update_predictions(revised_predictions)
            revision_prompts = []
            for prompt_index, prompt in enumerate(rev_dataset_obj.prompts):
                revision_prompt = prompt.replace('<extra_id_0>', revised_predictions[prompt_index])
                revision_prompt = revision_prompt.replace('<extra_id_1>', '').strip()
                revision_prompts.append(revision_prompt)
            rev_dataset_obj.setprompts(revision_prompts)

        # intermediate bleu score
        bleu_score = corpus_bleu(rev_dataset_obj.predictions, [dst_test_samples_subset]).score
        logging.debug("BLEU score: {}".format(bleu_score))
        # append the predictions to the predictions file
        with open(prediction_file, 'a') as f:
            f.write("\n".join(rev_dataset_obj.predictions))
            f.write("\n")
        # append the prompts to a prompt file
        with open(prompt_file, 'a') as f:
            f.write("\n".join(dataset_obj.prompts))
            f.write("\n")
        # append the revised prompts to a prompt file
        with open(revision_prompt_file, 'a') as f:
            f.write("\n".join(rev_dataset_obj.prompts))
            f.write("\n")
    # load the predictions from the predictions file
    with open(prediction_file, 'r') as f:
        predictions = f.read().splitlines()
    bleu_score = corpus_bleu(predictions, [dst_test_samples]).score
    logging.info("Final BLEU score: {}".format(bleu_score))
    return bleu_score


def predict_output(tokenizer, model, dataset_obj, revised=False, len_phrases_list=None, sent_index=None, extraction_pattern=None, fallback_pattern=None):
    predictions = []
    # run the model on the prompts in a batch to get the predictions
    batch = dataset_obj.prompts
    # replace \\n with newline characters in the batch
    batch = [prompt.replace("\\n", "\n") for prompt in batch]
    inputs = tokenizer(batch, padding=True, return_tensors="pt", max_length=2048).to("cuda")
    batch_predictions = model.generate(**inputs, max_length=50, do_sample=False, eos_token_id=2, early_stopping=True,
                             num_beams=5,
                             min_new_tokens=1)
    batch_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=False)

    for pred_index, pred in enumerate(batch_predictions):
        logging.debug(f"pred 0: {pred}, len(pred): {len(pred)}")
        match = extraction_pattern.search(pred)
        if match:
            extracted_text = match.group(1).replace("</s>", "").strip()
            logging.debug(f"pred extracted_text: {extracted_text}, len(extracted_text): {len(extracted_text)}")
        else:
            # extract text between <extra_id_0> and </s>
            match = fallback_pattern.search(pred)
            extracted_text = match.group(1).strip() if match else ""
            logging.debug(f"pred fallback: {extracted_text}, len(fallback): {len(extracted_text)}")

        if revised and sent_index > len_phrases_list[pred_index]:
            # set revised predictions to empty string if sent_index is longer than the length of the test sample
            extracted_text = ''

        predictions.append(extracted_text)

    return predictions


def construct_prompt(input_sample, source_lang='Hindi', target_lang='Marathi', prompt_template=None, chunk_size=-1):
    input_sample = split_phrases(input_sample, chunk_size)
    texts = [f"{prompt_template}\\n{source_lang}: {phrase}\\n{target_lang}: <extra_id_0>" for phrase in input_sample]
    return texts


def construct_revision_prompt(input_sample, prediction, previous_prompt, index, prev_revised_prediction, source_lang, target_lang, revision_template, prediction_list=None, chunk_size=-1):
    # include only the first four words of prediction
    prediction = " ".join(prediction.split(" ")[:4])

    text = revision_template
    input_sample = split_phrases(input_sample, chunk_size)
    if index == 0:
        text += f"\\n{source_lang}: {input_sample[index]} {input_sample[index + 1]}"
        text += f"\\n{target_lang}: <extra_id_0> {prediction} <extra_id_1>"
    elif index < len(input_sample) - 1:
        text += f"\\n{source_lang}: {input_sample[index - 1]} {input_sample[index]} {input_sample[index + 1]}"
        text += f"\\n{target_lang}: {prev_revised_prediction} <extra_id_0> {prediction} <extra_id_1>"
    elif index == len(input_sample) - 1:
        base_text = f"\\n\\nTranslate from {source_lang} to {target_lang}:"
        text = text.replace(base_text, "")
        text = text.replace(f"Translate from {source_lang} to {target_lang}:", "")
        text = text.replace(f"\\n{source_lang}:", base_text + f"\\n{source_lang}:")
        text += base_text
        # strip \\n\\n from the beginning of the text
        text = text[4:]
        text += f"\\n{source_lang}: {input_sample[index - 1]} {input_sample[index]}"
        text += f"\\n{target_lang}: {prev_revised_prediction} <extra_id_0>"
    else:
        text = previous_prompt
    return text

def split_phrases(input_sample, chunk_size):
    # split the input sample on spaces
    input_sample = input_sample.split()
    if len(input_sample) <= chunk_size:
        chunk_size = len(input_sample) - 1
    # Reverse the input sample
    input_sample.reverse()
    # Split the reversed input sample into chunks of size chunk_size
    input_sample = [input_sample[i:i + chunk_size][::-1] for i in range(0, len(input_sample), chunk_size)]
    # Reverse the list of chunks
    input_sample.reverse()
    # join the input sample with spaces
    input_sample = [" ".join(x) for x in input_sample]
    return input_sample


"""
Class to run the input samples in a batch
"""
class MyDataset(Dataset):
    def __init__(self):
        self.prompts = []
        self.predictions = []
        self.predictions_list = []

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def addprompt(self, prompt):
        self.prompts.append(prompt)

    def extendprompts(self, prompts):
        self.prompts.extend(prompts)

    def addprediction(self, prediction):
        self.predictions.append(prediction)

    def update_predictions(self, predictions):
        # iterate over the predictions
        for index, prediction in enumerate(predictions):
            # if index is greater than the length of the predictions list, then append the prediction to the list
            if index >= len(self.predictions):
                self.predictions.append(prediction)
                self.predictions_list.append([prediction])
            else:
                # update the prediction at the given index by appending the new prediction to the existing prediction separated by a space
                self.predictions[index] = self.predictions[index] + " " + prediction
                # strip the extra space at the end of the prediction
                self.predictions[index] = self.predictions[index].strip()
                # append the prediction to the predictions list
                self.predictions_list[index].append(prediction)

    # method to set the prompts
    def setprompts(self, prompts):
        self.prompts = prompts

    # method to set the predictions
    def setpredictions(self, predictions):
        self.predictions = predictions

    # set prompt at a given index
    def setprompt(self, index, prompt):
        self.prompts[index] = prompt


if __name__ == '__main__':
    # fetch the mode dev/test from the command line
    argparse = argparse.ArgumentParser()
    # fetch the model name from the command line
    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument("--mode", type=str, default="dev")
    # add argument for the input and output file
    argparse.add_argument("--input_file", type=str, default=None)
    argparse.add_argument("--output_file", type=str, default=None)
    # fetch the source and target languages from the command line
    argparse.add_argument("--source_language", type=str, default="hi")
    argparse.add_argument("--target_language", type=str, default="mr")
    # fetch the prompt template and revision template from the command line
    argparse.add_argument("--prompt_template", type=str, default=None, nargs='+')
    argparse.add_argument("--revision_template", type=str, default=None, nargs='+')
    # add argument for the batch size
    argparse.add_argument("--batch_size", type=int, default=1)
    # fetch the chunk size
    argparse.add_argument("--chunk_size", type=int, default=-1)
    # fetch the root output directory from the command line
    argparse.add_argument("--root_output_directory", type=str, default=None)
    args = argparse.parse_args()
    mode = args.mode
    if args.model_name == "xl":
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl", use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xl")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    elif args.model_name == "xxl":
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl", use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xxl", device_map="auto", torch_dtype=torch.float16)
    else:
        raise ValueError("Invalid model name")

    source_language = args.source_language
    target_language = args.target_language
    prompt_template = " ".join(args.prompt_template)
    revision_template = " ".join(args.revision_template)
    # iterate over the arguments and log them
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    # time the process
    start_time = time.time()
    process(tokenizer, model, mode, source_language, target_language, args.input_file, args.output_file, prompt_template, revision_template, args.batch_size, args.chunk_size, args.root_output_directory)
    end_time = time.time()
    # log the time taken in minutes
    logging.info(f"Time taken: {(end_time - start_time) / 60} minutes")
