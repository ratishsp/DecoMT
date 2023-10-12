import csv
import argparse
NEWLINE = "\\n"
import logging
logging.basicConfig(level=logging.INFO)

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
    'spa_Latn': 'Spanish',
    'por_Latn': 'Portuguese',
    'rus_Cyrl': 'Russian',
    'ukr_Cyrl': 'Ukrainian'
}

def process(input_file, is_source_first, source_lang, target_lang, output_dir, mode, prompt_chunk_size):
    # read the input csv file. The csv file has 10 rows and variable number of columns
    # each row represents a group of words
    # each column represents a word in the group
    # each row has a column with the length of the group
    # odd rows are the text in target language
    # even rows are the text in source language
    # there is no header in the csv file
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        # read all the rows
        rows = [row for row in reader]
        if is_source_first:
            # get the even rows
            source_text = rows[::2]
            # get the odd rows
            target_text = rows[1::2]
        else:
            # get the odd rows
            source_text = rows[1::2]
            # get the even rows
            target_text = rows[::2]
        updated_source_text = []
        updated_target_text = []
        # iterate through source and target text and trim the empty entries at the end
        for source_example, target_example in zip(source_text, target_text):
            # strip the two lists of empty entries at the end; start from the end
            while source_example and not source_example[-1]:
                source_example.pop()
            while target_example and not target_example[-1]:
                target_example.pop()
            # find the longer list; and pad the shorter list with empty entries
            if len(source_example) > len(target_example):
                target_example.extend([""] * (len(source_example) - len(target_example)))
            elif len(source_example) < len(target_example):
                source_example.extend([""] * (len(target_example) - len(source_example)))
            updated_source_text.append(source_example)
            updated_target_text.append(target_example)
        chunk_sizes = [5, 10]
        SOURCE = LANGUAGE_MAP[source_lang]
        TARGET = LANGUAGE_MAP[target_lang]
        TO_TARGET_ = f"{NEWLINE}{NEWLINE}Translate from %s to %s:" % (SOURCE, TARGET)
        TARGET_ = "%s%s: " % (NEWLINE, TARGET)
        SOURCE_ = "%s%s: " % (NEWLINE, SOURCE)
        templates = []
        for chunk_size in chunk_sizes:
            text = "Translate from %s to %s:" % (SOURCE, TARGET)
            # iterate through source and target text
            for source_example, target_example in zip(updated_source_text, updated_target_text):
                # source example is a list of words, target example is a list of words
                # we divide the list into sublists of length 5
                # each sublist represents a group of words
                source_example_ = [source_example[i:i + chunk_size] for i in range(0, len(source_example), chunk_size)]
                target_example_ = [target_example[i:i + chunk_size] for i in range(0, len(target_example), chunk_size)]
                logging.debug(f"Source Example {source_example_}")
                # handle the error "TypeError: not all arguments converted during string formatting"
                logging.debug(f"Target Example {target_example_}")
                # inspect each entry in the list. Each entry is of length 5. Find the index of the first non-empty entry
                # enumerate through the two lists and zip them together
                for example_index, (source_example_entry, target_example_entry) in enumerate(zip(source_example_, target_example_)):
                    logging.debug(f"Source Example Entry {source_example_entry}" )
                    logging.debug(f"Target Example Entry {target_example_entry}")
                    # get the index of the first non-empty entry
                    source_example_entry_index = next((i for i, x in enumerate(source_example_entry) if x), None)
                    target_example_entry_index = next((i for i, x in enumerate(target_example_entry) if x), None)
                    # if the index is not None, then we have found a non-empty entry
                    if source_example_entry_index is not None and target_example_entry_index is not None:
                        # find the higher index
                        example_entry_index = max(source_example_entry_index, target_example_entry_index)
                        # copy the entries [:source_example_entry_index] to the end of the previous entry
                        source_example_[example_index - 1] += source_example_entry[:example_entry_index]
                        target_example_[example_index - 1] += target_example_entry[:example_entry_index]
                        # remove the empty entries from the list
                        source_example_[example_index] = source_example_entry[example_entry_index:]
                        target_example_[example_index] = target_example_entry[example_entry_index:]
                    else:
                        source_example_[example_index - 1] += source_example_entry
                        target_example_[example_index - 1] += target_example_entry
                        source_example_[example_index] = []
                        target_example_[example_index] = []
                    # print the entry
                    logging.debug(" ".join(source_example_entry))
                    logging.debug(" ".join(target_example_entry))
                    logging.debug("")
                logging.debug("")
                for j, k in zip(source_example_, target_example_):
                    # if j and k are empty lists, then skip
                    if not j and not k:
                        continue
                    j = " ".join(j)
                    # strip empty spaces and replace multiple spaces with a single space
                    j = " ".join(j.split())
                    text += SOURCE_ + j
                    # add target word to the text
                    k = " ".join(k)
                    # strip empty spaces and replace multiple spaces with a single space
                    k = " ".join(k.split())
                    text += TARGET_ + k
                    logging.debug(j)
                    logging.debug(k)
                text += TO_TARGET_
            logging.debug(text)
            templates.append(text)

    batch_size = 5
    command = f"qsub -v OUTPUT_DIR=\"{output_dir}\",MODEL_NAME=\"xl\",MODE=\"{mode}\",BATCH_SIZE={batch_size},SOURCE_LANGUAGE=\"{source_lang}\",TARGET_LANGUAGE=\"{target_lang}\",CHUNK_SIZE={prompt_chunk_size},'prompt_template=\"{templates[0]}\"','revision_template=\"{templates[1]}\"' run_decomp_prompting.sh"
    print(command)


if __name__ == '__main__':
    # use argparse to parse the command line arguments
    parser = argparse.ArgumentParser()
    # read input file
    parser.add_argument('--source', type=str, required=True)
    # read boolean property is_source_first from the command line
    parser.add_argument('--is_source_first', action='store_true', help='set if source is first')
    # read source and target languages
    parser.add_argument('--source_language', type=str, required=True)
    parser.add_argument('--target_language', type=str, required=True)
    # read chunk size
    parser.add_argument('--chunk_size', type=int, required=True)
    # read output directory
    parser.add_argument('--output_dir', type=str, required=True)
    # read mode
    parser.add_argument('--mode', type=str, required=True)


    args = parser.parse_args()
    process(args.source, args.is_source_first, args.source_language, args.target_language, args.output_dir, args.mode, args.chunk_size)
