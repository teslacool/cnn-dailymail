import collections
import hashlib
import os
import io
import subprocess
import sys

# import tensorflow as tf
# from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

finished_files_dir = "finished_bert_files"

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

if len(sys.argv) != 3:
    print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
    sys.exit()
cnn_stories_dir = sys.argv[1]
dm_stories_dir = sys.argv[2]


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + "."


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    # abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])
    abstract = ' '.join(highlights)
    return article, abstract


def finalize(url_file, out_file, ):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    print("Making raw text file for URLs listed in %s..." % url_file)
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s + ".story" for s in url_hashes]
    num_stories = len(story_fnames)

    output_source = out_file + '.source'
    output_target = out_file + '.target'

    with io.open(output_source, 'w', newline='\n', encoding='utf8') as source_writer:
        with io.open(output_target, 'w', newline='\n', encoding='utf8') as target_writer:
            for idx, s in enumerate(story_fnames):
                if idx % 1000 == 0:
                    print("Writing story %i of %i; %.2f percent done" % (
                    idx, num_stories, float(idx) * 100.0 / float(num_stories)))

                # Look in the tokenized story dirs to find the .story file corresponding to this url
                if os.path.isfile(os.path.join(cnn_stories_dir, s)):
                    story_file = os.path.join(cnn_stories_dir, s)
                elif os.path.isfile(os.path.join(dm_stories_dir, s)):
                    story_file = os.path.join(dm_stories_dir, s)
                else:
                    print(
                        "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (
                        s, cnn_stories_dir, dm_stories_dir))
                    # Check again if tokenized stories directories contain correct number of files
                    print("Checking that the tokenized stories directories %s and %s contain correct number of files..." % (
                        cnn_stories_dir, dm_stories_dir))
                    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
                    check_num_stories(dm_stories_dir, num_expected_dm_stories)
                    raise Exception(
                        "Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (
                            cnn_stories_dir, dm_stories_dir, s))

                # Get the strings to write to .bin file
                article, abstract = get_art_abs(story_file)
                print(article, file=source_writer)
                print(abstract, file=target_writer)



    print("Finished writing articles into {} and abstracts into {}\n".format(output_source, output_target))




def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':

    # Check the stories directories contain the correct number of .story files
    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)



    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    finalize(all_test_urls, os.path.join(finished_files_dir, "test.bert"))
    finalize(all_val_urls, os.path.join(finished_files_dir, "val.bert"))
    finalize(all_train_urls, os.path.join(finished_files_dir, "train.bert"))
