import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
import logging
import argparse
# import statement for measuring time
import time
import cProfile

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
    'eng_Latn': 'English',
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


def get_samples(src_lang_code, dst_lang_code, mode):
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
        raise ValueError("mode should be dev or test")
    return src_test_samples, dst_test_samples


def process(tokenizer, model, mode, source_lang, target_lang, prompt_template, batch_size=5, root_output_directory=None):
    src_test_samples, dst_test_samples = get_samples(source_lang, target_lang, mode)
    # empty the prompts, predictions, and revision prompts files
    # create empty files for prompts with the part number
    prompt_file = os.path.join(root_output_directory, PROMPTS_TXT)
    prediction_file = os.path.join(root_output_directory, PREDICTIONS_TXT)
    open(prompt_file, 'w').close()
    open(prediction_file, 'w').close()
    for src_test_sample_index in range(0, len(src_test_samples), batch_size):
        dataset_obj = MyDataset()
        # get the prompts for the next 5 samples
        src_test_samples_subset = src_test_samples[src_test_sample_index:src_test_sample_index + batch_size]
        dst_test_samples_subset = dst_test_samples[src_test_sample_index:src_test_sample_index + batch_size]
        # construct the prompts for the next 5 samples
        source_lang_ = LANGUAGES[source_lang]
        target_lang_ = LANGUAGES[target_lang]
        for src_test_sample in src_test_samples_subset:
            prompt = construct_prompt(src_test_sample, source_lang=source_lang_, target_lang=target_lang_, prompt_template=prompt_template)
            dataset_obj.addprompt(prompt)
        # log the prompts which is a list of prompts
        logging.debug("dataset_obj.prompts %s", dataset_obj.prompts)
        # iterate through the indices in sent_index
        # the sent_index is an index of length of longest test sample
        # for each index, get the nth token from the src_test_samples
        # get the length of the longest test sample
        max_length = max([len(sample.split()) for sample in src_test_samples_subset])
        logging.info("max length: {}".format(max_length))
        # iterate till 1.5 times max_length
        for sent_index in range(int(1.5 * max_length)):
            predictions, is_eos = predict_output(tokenizer, model, dataset_obj, batch_size=batch_size, source_lang=source_lang_, target_lang=target_lang_)
            # method to write prompt to a prompt file
            logging.info(f"sent_index: {sent_index}")
            logging.info(f"predictions: {predictions[0]}")
            # print the nth token from the src_test_samples
            # handle the case where the nth token is not present
            update_prompts(dataset_obj, predictions, src_test_samples_subset, nth_token=sent_index, is_eos=is_eos)

        # replace <extra_id_0> with "" in the batch
        batch = [prompt.replace("<extra_id_0>", "") for prompt in dataset_obj.prompts]
        # find text after the last occurrence of {target_lang}: in the batch
        predictions = [prompt.split(f"{target_lang_}:")[-1] for prompt in batch]
        # strip the leading and trailing spaces from the predictions
        predictions = [prediction.strip() for prediction in predictions]

        bleu_score = corpus_bleu(predictions, [dst_test_samples_subset]).score
        logging.debug("BLEU score: {}".format(bleu_score))
        # append the predictions to the predictions file
        with open(prediction_file, 'a') as f:
            f.write("\n".join(predictions))
            f.write("\n")
        # append the prompts to a prompt file
        with open(prompt_file, 'a') as f:
            f.write("\n".join(dataset_obj.prompts))
            f.write("\n")
    # load the predictions from the predictions file
    with open(prediction_file, 'r') as f:
        predictions = f.read().splitlines()
    bleu_score = corpus_bleu(predictions, [dst_test_samples]).score
    logging.info("Final BLEU score: {}".format(bleu_score))
    return bleu_score


def update_prompts(dataset_obj, predictions, src_test_samples, nth_token=1, is_eos=None):
    updated_prompts = []
    for prompt_index, prompt in enumerate(dataset_obj.prompts):
        updated_prompt = prompt
        logging.debug(f"prompt: {prompt}")
        logging.debug(f"predictions[prompt_index]: {predictions[prompt_index]}, len(predictions[prompt_index]): {len(predictions[prompt_index])}")
        # strip " <extra_id_0>" from the prompt
        if is_eos[prompt_index]:
            if prompt.endswith(" <extra_id_0>"):
                updated_prompt = prompt[:-len(" <extra_id_0>")]
        else:
            updated_prompt = prompt.replace("<extra_id_0>", predictions[prompt_index])
            # append " <extra_id_0>" to the prompt
            updated_prompt = updated_prompt + " <extra_id_0>"
            if not predictions[prompt_index]:
                logging.info(f"predictions[prompt_index] is empty: {predictions[prompt_index]}")
        updated_prompts.append(updated_prompt)
    dataset_obj.setprompts(updated_prompts)


def predict_output(tokenizer, model, dataset_obj, batch_size=1, source_lang='Hindi', target_lang='Marathi'):
    predictions = []
    # run the model on the prompts in a batch to get the predictions
    batch = dataset_obj.prompts
    # replace \\n with newline characters in the batch
    batch = [prompt.replace("\\n", "\n") for prompt in batch]
    inputs = tokenizer(batch, padding=True, return_tensors="pt", max_length=2048).to("cuda")
    batch_predictions = model.generate(**inputs, max_length=10, do_sample=False, eos_token_id=2, early_stopping=True,
                             num_beams=5,
                             min_new_tokens=1)
    batch_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=False)
    # set a list of is_eos to indicate if the prediction {target_lang}:, {source_lang}:, </s> is present in the prediction
    # if the prediction is present then set the flag to True
    # else set the flag to False
    is_eos = []
    for pred_index, pred in enumerate(batch_predictions):
        # pred is of the format "<pad> <extra_id_0> उनके पास पहुंचने का प्रयास छोड़ दिया. Hindi: त्यांना</s> Marathi:</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s>"
        # need to extract the text after <extra_id_0> and before </s> or before "Hindi:" whichever comes first
        # split the string by "extra_id_0"
        pred = pred.split("<extra_id_0>")
        # get the second element of the list
        pred = pred[1]
        logging.debug(f"pred 1: {pred}, len(pred): {len(pred)}")
        pred = pred.split()[0]
        logging.debug(f"pred 3: {pred}, len(pred): {len(pred)}")
        if pred in ["Translate", f"{source_lang}:", f"{target_lang}:", "</s>"]:
            pred = ""
            is_eos.append(True)
        else:
            is_eos.append(False)
        predictions.append(pred)
    return predictions, is_eos


def construct_prompt(input_sample, source_lang='Hindi', target_lang='Marathi', prompt_template=None):
    text = prompt_template
    text += f"\\n{source_lang}: {input_sample}"
    text += f"\\n{target_lang}: <extra_id_0>"
    return text


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


if __name__ == '__main__':
    # fetch the mode dev/test from the command line
    argparse = argparse.ArgumentParser()
    # fetch the model name from the command line
    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument("--mode", type=str, default="dev")
    # fetch the source and target languages from the command line
    argparse.add_argument("--source_language", type=str, default="hi")
    argparse.add_argument("--target_language", type=str, default="mr")
    # fetch the prompt template and revision template from the command line
    argparse.add_argument("--prompt_template", type=str, default=None, nargs='+')
    # add argument for the batch size
    argparse.add_argument("--batch_size", type=int, default=1)
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
    # iterate over the arguments and log them
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    # time the process
    start_time = time.time()
    process(tokenizer, model, mode, source_language, target_language, prompt_template, args.batch_size, args.root_output_directory)
    end_time = time.time()
    # log the time taken in minutes
    logging.info(f"Time taken: {(end_time - start_time) / 60} minutes")
