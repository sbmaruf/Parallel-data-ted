"""
Python script for MT system.
"""
import os
import argparse
from collections import OrderedDict
import shutil
import numpy as np
import random
import sys
import hashlib

def _make_parser():
    parser = argparse.ArgumentParser(
                description="A script for generating train, dev and test for MT system.",
                prog=((sys.argv[2]+'.py') if os.path.basename(sys.argv[0]) == 'pydoc' else sys.argv[0]),
                formatter_class=argparse.RawDescriptionHelpFormatter
             )
    parser.add_argument("--dir",
                        default="./en-ms/",
                        type=str,
                        help="Directory of the data file.")
    parser.add_argument("--name",
                        default="./alt",
                        type=str,
                        help="Name of the final output file of NMT.")
    parser.add_argument("--src_lang",
                        default="en",
                        type=str,
                        help="Name of the log file")
    parser.add_argument("--tgt_lang",
                        default="ms",
                        type=str,
                        help="Name of the log file")
    parser.add_argument("--verbose",
                        default=1,
                        type=int,
                        help="Print extra information.")
    parser.add_argument("--seed",
                        default=1234,
                        type=int,
                        help="Seed for random number generation.")
    parser.add_argument("--distribution",
                        default='normal',
                        type=str,
                        help="Sampling distribution.")
    parser.add_argument("--dev_lines",
                        default='150 1000 800 800 300 300 300',
                        type=str,
                        help="How many lined will be taken for dev set from i'th dataset of dataset_name.")
    parser.add_argument("--test_lines",
                        default='100 500 500 500 200 200 200',
                        type=str,
                        help="How many lined will be taken for test set from i'th dataset of dataset_name.")
    parser.add_argument("--dataset_names",
                        default='alt amara os16 os18 ubuntu kde4 gnome',
                        type=str,
                        help="The dataset names.")
    parser.add_argument("--dev_percentage",
                        default='15 15 15 15 35 5',
                        type=str,
                        help="percentage of dev lines will be selected by each categorical function. "
                             "See the parse_params() function for details of the functions.")
    parser.add_argument("--test_percentage",
                        default='15 15 15 15 30 10',
                        type=str,
                        help="percentage of test lines will be selected by each categorical function. "
                             "See the parse_params() function for details of the functions.")
    parser.add_argument("--prod_test",
                        default=0,
                        type=int,
                        help="If you are testing reproduciblity or not. If this is set to 1 then this assumes"
                             "that there is already a dataset created and it creates a new dataset adding an extra `_`"
                             "with the each file name and do a `diff` test to see "
                             "if the dataset is successfully reproduced"
                             "or not.")
    return parser


def parse_params(params):

    """dev_lines, test_lines, dataset_names, dev_percentage, test_percentage
        carry the parallel information for each dataset in the dataset name

        Example of the parameters, dev_tot and test_tot is a dictionary

        dev_tot = {
            'alt': 150,
            'amara': 1000,
            'os16': 800,
            'os18': 800,
            'ubuntu': 300,
            'kde4': 300,
            'gnome': 300,
        }

        test_tot = {
            'alt': 100,
            'amara': 500,
            'os16': 500,
            'os18': 500,
            'ubuntu': 200,
            'kde4': 200,
            'gnome': 200,
        }

        percentage of sentences taken on the basis of

            (0). `num_of_token`
            (1). `num_of_token_diff`
            (2). `sum_of_token_len`
            (3). `sum_of_token_len_diff`
            (4). sum of total `freq_score` for each word
            (5). is there any out of `singleton` (single freq) in the line.

        `dev_percentage` and `test_percentage` is a list explaining
        what is the percentage of the `dev_tot[dataset]`/`test_tot[dataset]`
        will be taken from the function 0-5. each index of dev_percentage/test_percentage
        represent each funtion numbered above.

        example : dev_percentage[0] = 15, dev['alt'] = 150 means,
                    15 % of 150 = round(22.5) = 22 sentences will be selected
                    from function no (0) that is num_of_token.

        dev_percentage = [15, 15, 15, 15, 35, 5]
        test_percentage = [15, 15, 15, 15, 30, 10]

    """

    dev_percentage = [int(num) for num in params.dev_percentage.strip().split()]
    test_percentage = [int(num) for num in params.test_percentage.strip().split()]
    dataset_names = [name for name in params.dataset_names.strip().split()]
    dev_lines = [int(num) for num in params.dev_lines.strip().split()]
    test_lines = [int(num) for num in params.test_lines.strip().split()]
    try:
        assert len(dev_percentage) == len(test_percentage) == 6
        assert len(dataset_names) == len(dev_lines) == len(test_lines)
    except AssertionError:
        print("len(dev_percentage) :", len(dev_percentage))
        print("len(test_percentage) :", len(test_percentage))
        print("len(dataset_names) :", len(dataset_names))
        print("len(dev_lines) :", len(dev_lines))
        print("len(test_lines) :", len(test_lines))
        raise

    dev_tot = {}
    test_tot = {}

    suffix_add = '_' if params.prod_test == 1 else ''
    for i, name in enumerate(dataset_names):
        dev_tot[name+suffix_add] = dev_lines[i]
        test_tot[name+suffix_add] = test_lines[i]

    return (dev_percentage,
            test_percentage,
            dev_tot,
            test_tot)


def max_num_of_token(x, y):
    """
    sorting comparison function
    :param x: left iterator
    :param y: right iterator
    :return: x.num_of_token-y.num_of_token
    """
    if x.num_of_token == y.num_of_token:
        if x.num_of_token_diff == y.num_of_token_diff:
            if x.sum_of_token_len == y.sum_of_token_len:
                if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
                    if x.freq_score == y.freq_score:
                        return x.is_freq_one-y.is_freq_one
                    return y.freq_score - x.freq_score
                return x.sum_of_token_len_diff - y.sum_of_token_len_diff
            return x.sum_of_token_len - y.sum_of_token_len
        return x.num_of_token_diff - y.num_of_token_diff
    return x.num_of_token-y.num_of_token


def max_num_of_token_diff(x, y):
    """
        sorting comparison function
        :param x: left iterator
        :param y: right iterator
        :return: x.num_of_token_diff - y.num_of_token_diff
    """
    if x.num_of_token_diff == y.num_of_token_diff:
        if x.num_of_token == y.num_of_token:
            if x.sum_of_token_len == y.sum_of_token_len:
                if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
                    if x.freq_score == y.freq_score:
                        return x.is_freq_one-y.is_freq_one
                    return y.freq_score - x.freq_score
                return x.sum_of_token_len_diff - y.sum_of_token_len_diff
            return x.sum_of_token_len - y.sum_of_token_len
        return x.num_of_token-y.num_of_token
    return x.num_of_token_diff - y.num_of_token_diff


def sum_of_token_len(x, y):
    """
        sorting comparison function
        :param x: left iterator
        :param y: right iterator
        :return: x.sum_of_token_len - y.sum_of_token_len
    """
    if x.sum_of_token_len == y.sum_of_token_len:
        if x.num_of_token_diff == y.num_of_token_diff:
            if x.num_of_token == y.num_of_token:
                if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
                    if x.freq_score == y.freq_score:
                        return x.is_freq_one-y.is_freq_one
                    return y.freq_score - x.freq_score
                return x.sum_of_token_len_diff - y.sum_of_token_len_diff
            return x.num_of_token-y.num_of_token
        return x.num_of_token_diff - y.num_of_token_diff
    return x.sum_of_token_len - y.sum_of_token_len


def sum_of_token_len_diff(x, y):
    """
        sorting comparison function
        :param x: left iterator
        :param y: right iterator
        :return: x.sum_of_token_len_diff - y.sum_of_token_len_diff
    """
    if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
        if x.num_of_token_diff == y.num_of_token_diff:
            if x.sum_of_token_len == y.sum_of_token_len:
                if x.num_of_token == y.num_of_token:
                    if x.freq_score == y.freq_score:
                        return x.is_freq_one-y.is_freq_one
                    return y.freq_score - x.freq_score
                return x.num_of_token-y.num_of_token
            return x.sum_of_token_len - y.sum_of_token_len
        return x.num_of_token_diff - y.num_of_token_diff
    return x.sum_of_token_len_diff - y.sum_of_token_len_diff


def freq_score(x, y):
    """
        sorting comparison function
        :param x: left iterator
        :param y: right iterator
        :return: y.freq_score - x.freq_score
    """
    if x.freq_score == y.freq_score:
        if x.num_of_token_diff == y.num_of_token_diff:
            if x.sum_of_token_len == y.sum_of_token_len:
                if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
                    if x.num_of_token == y.num_of_token:
                        return x.is_freq_one-y.is_freq_one
                    return x.num_of_token-y.num_of_token
                return x.sum_of_token_len_diff - y.sum_of_token_len_diff
            return x.sum_of_token_len - y.sum_of_token_len
        return x.num_of_token_diff - y.num_of_token_diff
    return y.freq_score - x.freq_score


def singleton(x, y):
    """
        sorting comparison function
        :param x: left iterator
        :param y: right iterator
        :return: x.is_freq_one-y.is_freq_one
    """
    if x.freq_score == y.freq_score:
        if x.num_of_token_diff == y.num_of_token_diff:
            if x.sum_of_token_len == y.sum_of_token_len:
                if x.sum_of_token_len_diff == y.sum_of_token_len_diff:
                    if x.num_of_token == y.num_of_token:
                        return x.num_of_token - y.num_of_token
                    return y.freq_score - x.freq_score
                return x.sum_of_token_len_diff - y.sum_of_token_len_diff
            return x.sum_of_token_len - y.sum_of_token_len
        return x.num_of_token_diff - y.num_of_token_diff
    return x.is_freq_one-y.is_freq_one


def cmp_to_key(mycmp):
    """
    class for sorting comparison function
    :param mycmp: a LinePair type class
    :return: a class comprised of necessary operators for sorting comparison function
    """
    class K(object):
        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


class LinePair:
    """
    A class for saving two line pair and their attribute.
    attributes,
    0. src_line : a parallel line of a source language.
    1. tgt_line : a parallel line of a target language.
    2. id: line pair id
    3. src_words : src_line words in a list splited by ' '
    4. tgt_words : tgt_line words in a list splited by ' '
    5. num_of_token : total number of token of two lines.
    6. num_of_token_diff : number of token differece of two language.
    7. sum_of_token_len : total number of characters. (sum of token lengths)
    8. sum_of_token_len_diff : total number of token lengths difference.
    9. freq_score = sum of the frequecy of the words in src_line and tgt_line
    10. is_freq_one = if there is any word whose frequency is one or not.
    """
    def __init__(self, _src_line, _tgt_line):
        self.src_line = _src_line
        self.tgt_line = _tgt_line
        self.id = _src_line + _tgt_line
        self.src_words = _src_line.strip().split()
        self.tgt_words = _tgt_line.strip().split()
        self.num_of_token = len(self.src_words) + len(self.tgt_words)
        self.num_of_token_diff = abs(len(self.src_words) - len(self.tgt_words))
        self.sum_of_token_len = (sum(list(map(len, self.src_words))) +
                                 sum(list(map(len, self.tgt_words))))
        self.sum_of_token_len_diff = abs(sum(list(map(len, self.src_words))) -
                                         sum(list(map(len, self.tgt_words))))
        self.freq_score = 0
        self.is_freq_one = 0


def retrieve_file_dict(_all_files, src_lang, tgt_lang):
    """
    retrieve the filename form a nmt based naming files.
    example:
        input : gnome.en-ms.en
        output : gnome

    :param _all_files: the nmt bas named file in a list.
    :param src_lang: the src_language short form.
    :param tgt_lang: the tgt_language short form.
    :return: a dictionary(OrderedDict) of file names.
    """
    search_str = '.' + src_lang + '-' + tgt_lang
    src_search_str = search_str + '.' + src_lang
    tgt_search_str = search_str + '.' + tgt_lang

    _file_dict = OrderedDict()
    for files in _all_files:
        if src_search_str in files:
            file_name = files.split(sep=src_search_str)
            _file_dict[file_name[0]] = 1
        if tgt_search_str in files:
            file_name = files.split(sep=tgt_search_str)
            _file_dict[file_name[0]] = 1
    return _file_dict


def select_line_ids(folder_name,
                    tot_line_selected,
                    total_num_lines):
    """
    Function for select lines from a sorted array. (sorted based of 6 attribute written in parse params)
    :param folder_name: is folder name is `test` then it do some more procesing to make it harder than `dev`
    :param tot_line_selected: total number of index to be selected.
    :param total_num_lines: the domain of the range. total number of line exists in the dataset. [0:total_num_lines]
    :return: selected line ids.
    TODO: implement for both normal and uniform
    """

    indexes = list(np.linspace(0, total_num_lines,
                               num=tot_line_selected,
                               endpoint=False))

    if folder_name == 'test':
        indexes = map(lambda x: x+5, indexes)

    indexes = set(list(map(round, indexes)))
    index_out_of_boundary = 0
    for index in indexes:
        if index >= total_num_lines:
            index_out_of_boundary += 1
            indexes.remove(index)

    # to ensure test set is harder than dev set
    if folder_name == 'test' and index_out_of_boundary > 0:
        idx = total_num_lines-1
        while index_out_of_boundary > 0:
            if idx not in indexes:
                indexes.add(idx)
                index_out_of_boundary -= 1

    # to ensure fructional problem doesn't
    # hamper selecting total_num_lines
    idx = 0
    while len(indexes) != tot_line_selected:
        if idx == total_num_lines:
            break
        if idx not in indexes:
            indexes.add(idx)
        idx += 1
    return list(indexes)


def write_data(par_line,
               file_name,
               folder_name,
               percentage,
               folder_address,
               comp_funcs,
               total_sent_target,
               id_set=None):
    """
    write parallel lines on a file based on some attribute.
    :param par_line: a list of LinePair obj.
    :param file_name: the dataset name
    :param folder_name: either `dev` or `test`. Indicates if I am writting for dev set of test set.
    :param percentage: the percentage of lines will be taken by each function of `comp_funcs`. the i'th percentage is
                        for i'th `comp_funcs`
    :param comp_funcs: the list of comparison function which will be used to sort LinePair objects.
    :param folder_address: the folder address where the new `dev` ot `test` set of the dataset will be created.
    :param total_sent_target: total number of lines to be retrieved.
    :param id_set: the line id that should not be taken.
    :return:
    """

    try:
        assert folder_name == 'dev' \
               or folder_name == 'test' \
               or folder_name == 'train'
    except AssertionError:
        print("folder_name :", folder_name)
        raise

    try:
        assert len(percentage) == len(comp_funcs)
    except AssertionError:
        print("len(percentage) :", len(percentage))
        print("len(comp_funcs) :", len(comp_funcs))
        raise

    file_address = os.path.join(folder_address, file_name) + \
                   '.' + folder_name + \
                   '.' + params.src_lang + '-' + params.tgt_lang
    src_file_address = file_address + '.' + params.src_lang
    tgt_file_address = file_address + '.' + params.tgt_lang

    total_num_lines = len(par_line)
    total_sent_writen = 0
    with open(src_file_address, "w") as src_file_ptr, \
            open(tgt_file_address, "w") as tgt_file_ptr:

        index_taken = set() if id_set is None else id_set

        if folder_name == 'train':
            for line_pair in par_line:
                if line_pair.id not in index_taken:
                    _src_line = line_pair.src_line
                    _tgt_line = line_pair.tgt_line
                    if _src_line == "" or _tgt_line == "":
                        continue
                    src_file_ptr.write(_src_line)
                    tgt_file_ptr.write(_tgt_line)

                    total_sent_writen += 1
                    index_taken.add(line_pair.id)

        # for dev and test
        else:
            try:
                assert total_sent_target <= total_num_lines
            except AssertionError:
                print("folder_name :", folder_name)
                print("file_name :", file_name)
                print("Requested lines :", total_sent_target)
                print("Total number of line exists :", total_num_lines)
                raise

            _idx = 0
            for comp_func in comp_funcs:

                if params.verbose == 1:
                    print("\t(", _idx, "). Function name: ", comp_func.__name__)

                sorted_par_lines = sorted(par_line,
                                          key=cmp_to_key(comp_func))

                tot_line_selected = round((total_sent_target * percentage[_idx]) / 100)
                if params.verbose == 1:
                    print("\t\t", percentage[_idx], "% of",
                          total_sent_target, "is round(",
                          (total_sent_target * percentage[_idx]) / 100, ") =",
                          tot_line_selected)

                indexes = select_line_ids(folder_name,
                                          tot_line_selected,
                                          len(sorted_par_lines))

                if params.verbose == 1:
                    print("\t\tTotal line selected : ", len(indexes))

                tot_line_each_case = 0
                for i in indexes:
                    if sorted_par_lines[int(i)].id not in index_taken:

                        _src_line = sorted_par_lines[int(i)].src_line
                        _tgt_line = sorted_par_lines[int(i)].tgt_line
                        if _src_line == "" or _tgt_line == "":
                            continue

                        src_file_ptr.write(_src_line)
                        tgt_file_ptr.write(_tgt_line)

                        index_taken.add(sorted_par_lines[int(i)].id)
                        tot_line_each_case += 1
                        total_sent_writen += 1

                if params.verbose == 1:
                    print("\t\tTotal line written : ", tot_line_each_case)
                _idx += 1

            while total_sent_writen < total_sent_target:

                rand_index = np.random.randint(0, total_num_lines, size=1)

                if par_line[int(rand_index)].id not in index_taken:

                    _src_line = par_line[int(rand_index)].src_line
                    _tgt_line = par_line[int(rand_index)].tgt_line

                    if _src_line == "" or _tgt_line == "":
                        continue

                    src_file_ptr.write(_src_line)
                    tgt_file_ptr.write(_tgt_line)

                    index_taken.add(par_line[int(rand_index)].id)
                    total_sent_writen += 1

    return index_taken, total_sent_writen


def copy_files(folder_name, file_dict, src_lang, tgt_lang):
    """
    Copy files from a folder giving the base folder name and dataset and language name.
    :param folder_name: the base_folder name.
    :param file_dict: the dictionary of files.
    :param src_lang: short form of the src language
    :param tgt_lang: short form of the tgt language
    :return: all the copied files address list.
    """
    search_str = '.' + src_lang + '-' + tgt_lang
    src_search_str = search_str + '.' + src_lang
    tgt_search_str = search_str + '.' + tgt_lang

    dataset_address_list = []
    for i, j in file_dict.items():

        os.makedirs(os.path.join(folder_name, i), exist_ok=True)

        src_file_address = os.path.join(folder_name, i) + src_search_str
        tgt_file_address = os.path.join(folder_name, i) + tgt_search_str

        new_src_file_address = os.path.join(
                                    os.path.join(folder_name, i),
                                    i + src_search_str)
        new_tgt_file_address = os.path.join(
                                    os.path.join(folder_name, i),
                                    i + tgt_search_str)

        dataset_address_list.append((new_src_file_address,
                                     new_tgt_file_address))

        try:
            shutil.copy(src_file_address, new_src_file_address)
        except FileNotFoundError as e:
            print(e)
            pass
        try:
            shutil.copy(tgt_file_address, new_tgt_file_address)
        except FileNotFoundError as e:
            print(e)
            pass
    return dataset_address_list


def extract_lines(src_data_add, tgt_data_add):
    """
    Extract LinePair object by giving two parallel file address.
    also update the frequency attribute of a LinePair object.
    :param src_data_add: address of the source file.
    :param tgt_data_add: address of the target file.
    :return: list of LinePair object read from the src_data_add and tgt_data_add, a freq dictionary.
    """
    src_num_of_line = sum(1 for _ in open(src_data_add))
    tgt_num_of_line = sum(1 for _ in open(tgt_data_add))
    try:
        assert src_num_of_line == tgt_num_of_line
    except AssertionError:
        print("src_num_of_line :", src_num_of_line)
        print("tgt_num_of_line :", tgt_num_of_line)
        raise

    idx = 0
    par_line = []
    frq = {}
    with open(src_data_add) as src_ptr, \
            open(tgt_data_add) as tgt_ptr:
        for (src_line, tgt_line) in zip(src_ptr, tgt_ptr):
            par_line.append(LinePair(src_line, tgt_line))
            for i in par_line[idx].src_words:
                if i not in frq:
                    frq[i] = 0
                else:
                    frq[i] += 1
            for i in par_line[idx].tgt_words:
                if i not in frq:
                    frq[i] = 0
                else:
                    frq[i] += 1
            idx += 1
    return par_line, frq


def augment_freq(par_line, frq):
    """
    Update the `freq_score` and `is_freq_one` attribute of each `LinePair` objects that is in the `par_line` list
    :param par_line: a list consists of LinePair object
    :param frq: a dictionary of consists of the token frequency.
    :return: the updated `par_line` with new `freq_score` and 'is_freq_one`, total_number of sungleton pair exists.
    """
    total_singleton = 0
    tot_lines = len(par_line)
    for i in range(tot_lines):
        is_singleton_pair = 0
        for word in par_line[i].src_words:
            frq_word = frq[word]
            par_line[i].freq_score += frq_word
            if frq_word == 1:
                par_line[i].is_freq_one = 1
                is_singleton_pair = 1
        for word in par_line[i].tgt_words:
            frq_word = frq[word]
            par_line[i].freq_score += frq_word
            if frq_word == 1:
                par_line[i].is_freq_one = 1
                is_singleton_pair = 1
        total_singleton += is_singleton_pair
    return par_line, total_singleton


def write_test(_par_line, src_data_add, tgt_data_add):
    """
    testing code to see if file read write doesn't destroy any data.
    Sometimes encoding like utf-8, latin-1 may conflict.
    :param _par_line: a list of `LinePair` object.
    :param src_data_add: address from where source file will be read.
    :param tgt_data_add: address from where target file will be read.
    :return: return 0 on success.

    """
    dummy_src_data_add = src_data_add + '.temp'
    dummy_tgt_data_add = tgt_data_add + '.temp'
    with open(dummy_src_data_add, "w") as src_ptr, \
            open(dummy_tgt_data_add, "w") as tgt_ptr:
        for i in range(len(_par_line)):
            src_ptr.write(_par_line[i].src_line)
            tgt_ptr.write(_par_line[i].tgt_line)

    try:
        cmd_src = "diff " + src_data_add + " " + dummy_src_data_add
        assert os.system(cmd_src) == 0
    except AssertionError as e:
        print("src_data_add: ", src_data_add)
        print("dummy_src_data_add: ", dummy_src_data_add)
        print(e)
        raise

    try:
        cmd_tgt = "diff " + tgt_data_add + " " + dummy_tgt_data_add
        assert os.system(cmd_tgt) == 0
    except AssertionError as e:
        print("tgt_data_add: ", tgt_data_add)
        print("dummy_tgt_data_add: ", dummy_tgt_data_add)
        print(e)
        raise

    cmd_src = "rm " + dummy_src_data_add
    cmd_tgt = "rm " + dummy_tgt_data_add
    os.system(cmd_src)
    os.system(cmd_tgt)
    return 0


def create_dirs(src_data_add):
    """
    Create directory for train, dev and test
    :param src_data_add: the address where the folders will be created.
    :return: the address of the new train, dev and test folder.
    """
    new_dev_folder_address = os.path.join(
        os.path.dirname(src_data_add),
        'dev')
    new_test_folder_address = os.path.join(
        os.path.dirname(src_data_add),
        'test')
    new_train_folder_address = os.path.join(
        os.path.dirname(src_data_add),
        'train')

    os.makedirs(new_dev_folder_address, exist_ok=True)
    if params.verbose == 1:
        print("New folder created at :", new_dev_folder_address)
    os.makedirs(new_test_folder_address, exist_ok=True)
    if params.verbose == 1:
        print("New folder created at :", new_test_folder_address)
    os.makedirs(new_train_folder_address, exist_ok=True)
    if params.verbose == 1:
        print("New folder created at :", new_train_folder_address)

    return (new_dev_folder_address,
            new_test_folder_address,
            new_train_folder_address)


def index_list_test(par_line, index_set):
    """
    testing function. after creating all train, dev and test
    this function ensures that all the lines are written at least one the train, dev and test dataset.
    :param par_line: a list of LinePair Object.
    :param index_set: the set of index taken by train, dev or test.
    :return: return 0 on success.
    """
    for i in range(len(par_line)):
        try:
            assert par_line[i].id in index_set
        except AssertionError:
            print("par_line[i].id :", par_line[i].id)
            raise
    return 0


def reproduciblity_test(dummy_file_name,
                        folder_name,
                        base_folder_address):
    """
    testing reproduciblity of the dataset. this assumes that we already have a cloned file in earlier run
    and current run creates a new files adding '_' at the end the each file name. now it's comparing two file
    with a system `diff` call.
    :param dummy_file_name: the dataset name + '_'. This is created to test reproduciblity.
    :param folder_name: the folder where cloned dataset exists. it will be `train`, `dev` or `test`
    :param base_folder_address: the base folder where the dataset exists.
    :return: return 0 on success.
    """
    dummy_file_address = os.path.join(base_folder_address, dummy_file_name) + \
                         '.' + folder_name + \
                         '.' + params.src_lang + '-' + params.tgt_lang
    dummy_src_file_address = dummy_file_address + '.' + params.src_lang
    dummy_tgt_file_address = dummy_file_address + '.' + params.tgt_lang

    file_name = dummy_file_name[0:-1]
    file_address = os.path.join(base_folder_address, file_name) + \
                   '.' + folder_name + \
                   '.' + params.src_lang + '-' + params.tgt_lang
    src_file_address = file_address + '.' + params.src_lang
    tgt_file_address = file_address + '.' + params.tgt_lang

    cmd_src = "diff " + src_file_address + " " + dummy_src_file_address
    cmd_tgt = "diff " + tgt_file_address + " " + dummy_tgt_file_address

    try:
        assert os.system(cmd_src) == 0
    except AssertionError:
        print(cmd_src)
        raise

    try:
        assert os.system(cmd_tgt) == 0
    except AssertionError:
        print(cmd_tgt)
        raise

    cmd_src = "rm " + dummy_src_file_address
    cmd_tgt = "rm " + dummy_tgt_file_address

    os.system(cmd_src)
    os.system(cmd_tgt)
    return 0


###################################
# Start of the python script
###################################

def main(params):

    comp_funcs = [max_num_of_token,
                  max_num_of_token_diff,
                  sum_of_token_len,
                  sum_of_token_len_diff,
                  freq_score,
                  singleton]

    (dev_percentage,
     test_percentage,
     dev_tot,
     test_tot) = parse_params(params)

    if params.verbose == 1:
        print("Arg values")
        print(params)
        print("-"*100)
        print("`dev_tot` dictionary contains all the dataset name and "
              "the number of line we need to retrieve for dev set.")
        print("Current dev_tot dictionary.")
        for k, v in dev_tot.items():
            print(k, ":", v)
        print("`test_tot` dictionary contains all the dataset name and "
              "the number of line we need to retrieve for test set.")
        print("Current test_tot dictionary.")
        for k, v in test_tot.items():
            print(k, ":", v)
        print("-"*100)

    # for reproduciblity
    random.seed(params.seed)
    np.random.seed(params.seed)

    all_files = os.listdir(params.dir)
    all_files.sort(key=lambda x: x.lower())

    file_dict = retrieve_file_dict(all_files,
                                   params.src_lang,
                                   params.tgt_lang)

    if params.verbose == 1:
        print("\nName of the all dataset.")
        for k, v in test_tot.items():
            print(k, end=" ")
        print("\n")

    dataset_address_list = copy_files(params.dir,
                                      file_dict,
                                      params.src_lang,
                                      params.tgt_lang)

    for src_data_add, tgt_data_add in dataset_address_list:

        file_name = str(os.path.basename(src_data_add).split('.')[0])
        if params.prod_test == 1:
            file_name += "_"
        if params.verbose == 1:
            print("#"*100)

        print("Working on {} dataset.".format(file_name))

        if params.verbose == 1:
            print("#"*100)

        par_line_, frq = extract_lines(src_data_add, tgt_data_add)

        # see if there is any information loss for python read and write
        write_test(par_line_, src_data_add, tgt_data_add)
        par_line, singleton_pair = augment_freq(par_line_, frq)

        try:
            assert len(par_line) == len(par_line_)
            par_line_ = []  # clear memory
        except AssertionError:
            print("par_line", len(par_line))
            print("par_line_", len(par_line_))
            raise

        if params.verbose == 1:
            print("Total number of extracted line", len(par_line))
            print("Total number of line pair where singleton exists.",
                  singleton_pair)

        total_num_of_lines = len(par_line)

        (new_dev_folder_address,
         new_test_folder_address,
         new_train_folder_address) = create_dirs(src_data_add)

        if params.verbose == 1:
            print("\n\nDEV DATASET CREATION")
            print("-"*30)

        total_sent_written_train_dev_test = 0
        (sentence_taken,
         total_sent_writen) = write_data(par_line,
                                         file_name,
                                         'dev',
                                         dev_percentage,
                                         new_dev_folder_address,
                                         comp_funcs,
                                         dev_tot[file_name],
                                         id_set=None)
        if params.prod_test == 1:
            reproduciblity_test(file_name,
                                'dev',
                                new_dev_folder_address)

        total_sent_written_train_dev_test += total_sent_writen

        if params.verbose == 1:
            print("Total number of dev sentence requested :",
                  dev_tot[file_name])
            print("Total number of dev sentences taken :",
                  total_sent_writen)

        if params.verbose == 1:
            print("\n\nTEST DATASET CREATION")
            print("-"*30)

        (dev_test_index,
         total_sent_writen) = write_data(par_line,
                                         file_name,
                                         'test',
                                         test_percentage,
                                         new_test_folder_address,
                                         comp_funcs,
                                         test_tot[file_name],
                                         id_set=sentence_taken)
        if params.prod_test == 1:
            reproduciblity_test(file_name,
                                'test',
                                new_test_folder_address)

        total_sent_written_train_dev_test += total_sent_writen

        if params.verbose == 1:
            print("Total number of test sentence requested :",
                  test_tot[file_name])
            print("Total number of test sentences taken :",
                  total_sent_writen)

        if params.verbose == 1:
            print("\n\nTRAIN DATASET CREATION")
            print("-"*30)

        (index_set,
         total_sent_writen) = write_data(par_line,
                                         file_name,
                                         'train',
                                         [100],
                                         new_train_folder_address,
                                         [None],
                                         None,
                                         id_set=dev_test_index)

        if params.prod_test == 1:
            reproduciblity_test(file_name,
                                'train',
                                new_train_folder_address)

        total_sent_written_train_dev_test += total_sent_writen
        index_list_test(par_line, index_set)

        if params.verbose == 1:
            print("Total number of train sentences taken :",
                  total_sent_writen)

        try:
            assert total_sent_written_train_dev_test == total_num_of_lines
        except AssertionError:
            print("total_sent_written_train_dev_test :", total_sent_written_train_dev_test)
            print("total number of line in the file :", len(par_line))
            if len(par_line) > total_sent_written_train_dev_test:
                tot_missing_line = (len(par_line) - total_sent_written_train_dev_test)
                temp_dict = set()
                try:
                    for line_pair in par_line:
                        src_tgt = line_pair.id
                        temp_dict.add(src_tgt)
                    missing_line_calc = len(par_line) - len(temp_dict)
                    assert missing_line_calc == tot_missing_line
                    print("{0} number of repetitive line in {1} dataset".
                          format(tot_missing_line, file_name))
                except AssertionError:
                    print("total number of unique line in the dataset {0}".format(len(temp_dict)))
                    print("line missing :", tot_missing_line)
                    raise
            else:
                print("line overwritten", total_sent_written_train_dev_test-len(par_line))
                raise

        if params.verbose == 1:
            print("-"*100, "\n")


parser = _make_parser()
__doc__ += parser.format_help()

if __name__ == "__main__":
    params = parser.parse_args()
    main(params)
