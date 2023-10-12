# script to create prompt for few shot sap
import argparse
import os

NEWLINE = "\\n"

# map from language code to language name
LANGUAGE_MAP = {
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


def process(model_name, source_lang_, target_lang_, flores_dev_set, output_dir, mode):
    # read the first five lines of the flores dev set file for the source language
    # read the next five lines of the flores dev set file for the target language
    # iterate through source and target text
    # append the source and target text to the prompt template
    source_file = os.path.join(flores_dev_set, f"{source_lang_}.dev")
    target_file = os.path.join(flores_dev_set, f"{target_lang_}.dev")
    source_lang = LANGUAGE_MAP[source_lang_]
    target_lang = LANGUAGE_MAP[target_lang_]
    prompt_template = f"Translate from {source_lang} to {target_lang}:"
    with open(source_file, 'r') as f:
        source_text = f.readlines()[:5]
    with open(target_file, 'r') as f:
        target_text = f.readlines()[:5]
    for source_example, target_example in zip(source_text, target_text):
        prompt_template += f"{NEWLINE}{source_lang}: {source_example.strip()}"
        prompt_template += f"{NEWLINE}{target_lang}: {target_example.strip()}"
        prompt_template += f"{NEWLINE}{NEWLINE}Translate from {source_lang} to {target_lang}:"
    # escape the double quotes in the prompt template
    prompt_template = prompt_template.replace("\"", "\\\"")
    # escape the single quotes in the prompt template
    prompt_template = prompt_template.replace("\'", "\\\'")

    command = f"qsub -v OUTPUT_DIR=\"{output_dir}\",MODEL_NAME=\"{model_name}\",MODE=\"{mode}\",BATCH_SIZE=5,SOURCE_LANG=\"{source_lang_}\",TARGET_LANG=\"{target_lang_}\",'prompt_template=\"{prompt_template}\"' run_fewshot_baseline.sh"
    return command

if __name__ == "__main__":
    # using argparse to parse the command line arguments; fetch the source and target language and path of flores dev set
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_language", type=str, required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument("--flores_dev_set", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    # model name
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    command = process(args.model_name, args.source_language, args.target_language, args.flores_dev_set, args.output_dir, args.mode)
    print(command)
