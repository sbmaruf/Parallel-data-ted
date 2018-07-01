import os
import argparse
import time
import sys
from collections import OrderedDict
import re
import unicodedata

parser = argparse.ArgumentParser(
            description="Accumulate different source of data",
            prog=((sys.argv[2] + '.py') if os.path.basename(sys.argv[0]) == 'pydoc' else sys.argv[0]),
            formatter_class=argparse.RawDescriptionHelpFormatter
         )
parser.add_argument("--src_lang",
                    type=str,
                    default="en",
                    help="short identifier of the source language. ex: en")
parser.add_argument("--tgt_lang",
                    type=str,
                    default="ms",
                    help="short identifier of the target language. ex: ms")
parser.add_argument("--dir",
                    type=str,
                    default="./en-ms/",
                    help="folder address where the data files exists. data file pattern:"
                         " file_name.lang1-lang2.selected_lang. ex: amara.en-ms.en")
parser.add_argument("--out_dir",
                    type=str,
                    default="./nmt_io/",
                    help="folder address where the data files will be saved.")
parser.add_argument("--parallel_file",
                    type=int,
                    default=1,
                    help="if the parallel lines will be saved in same file or not."
                         "condition for parallel_file == 0 is not implemented in the script")
parser.add_argument("--debug",
                    type=int,
                    default=1,
                    help="if the parallel lines will be saved in same file or not.")
parser.add_argument("--restrict",
                    type=str,
                    default="",
                    help="if the parallel lines will be saved in same file or not.")
parser.add_argument("--verbose",
                    type=int,
                    default=2,
                    help="printing additional information."
                    "verbose = 2 : show the details step of folder creation."
                    "verbose = 1 : Additional information printing.")


def retrieve_file_dict(_all_files, src_lang, tgt_lang):
    """
    retrieve the filename form a nmt based naming files.
    example:
        input : gnome.en-ms.en
        output : gnome

    :param _all_files: the nmt bas named file in a list.
    :param src_lang: the src_language short form.
    :param tgt_lang: the tgt_language short form.
    :return: a dictionary(OrderedDict) of file names, and the output_file_name in string
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

    output_file_name = ''
    flag = 0
    for name in _file_dict:
        output_file_name = output_file_name + ("_" if flag > 0 else "") + name
        flag = 1
    return _file_dict, output_file_name


def assert_check_before(_src_file_address,
                        _tgt_file_address):

    _num_lines_src = sum(1 for _ in open(_src_file_address))
    _num_lines_tgt = sum(1 for _ in open(_tgt_file_address))
    line_no = 0
    for src_line, tgt_line in zip(open(_src_file_address),
                                  open(_tgt_file_address)):
        line_no += 1
        try:
            assert src_line != "" and tgt_line != ""
        except AssertionError:
            print("Empty line found in {0}, {1} at line {2}.".
                  format(_src_file_address, _tgt_file_address, line_no))
    try:
        assert _num_lines_src == _num_lines_tgt
    except AssertionError:
        print("Before reading dataset testing lines equality.")
        print("num_lines_src :", _num_lines_src)
        print("num_lines_tgt :", _num_lines_tgt)
        print("Total loss :", abs(_num_lines_src - _num_lines_tgt))
        raise

    return _num_lines_src, _num_lines_tgt


def assert_check_after(_num_lines_src,
                       _num_lines_tgt,
                       _dummy_output_file_address_src,
                       _dummy_output_file_address_tgt,
                       _tot_num_of_line,
                       parallel_file=1,
                       skipped_lines=0):
    if parallel_file:
        num_lines_src_output = sum(1 for _ in open(_dummy_output_file_address_src))
        num_lines_tgt_output = sum(1 for _ in open(_dummy_output_file_address_tgt))
        try:
            assert num_lines_src_output == num_lines_tgt_output == \
                   _tot_num_of_line + _num_lines_src - skipped_lines
        except AssertionError:
            print("Testing after adding lines to output files. (parallel_data=1)")
            print("num_lines_src :", _num_lines_src)
            print("num_lines_tgt :", _num_lines_tgt)
            print("num_lines_src_output :", num_lines_src_output)
            print("num_lines_tgt_output :", num_lines_tgt_output)
            print("tot_num_of_line :", _tot_num_of_line)
            print("_tot_num_of_line + _num_lines_src:", _tot_num_of_line + _num_lines_src)
            print("Total loss :", abs(num_lines_src_output - num_lines_tgt_output))
            raise
    else:
        raise Exception("condition for parallel_file == 0 is not implemented in the script")


def assert_check_inside(_dummy_output_file_address_src,
                        _dummy_output_file_address_tgt,
                        _tot_num_of_line,
                        _cnt):

    num_line_src = sum(1 for _ in open(_dummy_output_file_address_src))
    num_line_tgt = sum(1 for _ in open(_dummy_output_file_address_tgt))
    try:
        assert (_tot_num_of_line + _cnt) == num_line_src == num_line_tgt
    except AssertionError:
        print("tot_num_of_line :", _tot_num_of_line)
        print("cnt :", _cnt)
        print("num_line_src :", num_line_src)
        print("num_line_tgt :", num_line_tgt)
        raise


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def refine(par_line):
    """
    TODO: If there is any pre-processing needed before accumulating all data
    :param par_line: a string that will be refined
    :return: ret: the refined string
    """
    ret = unicode_to_ascii(par_line)
    return ret


def write_lines(_src_file_address,
                _tgt_file_address,
                _dummy_output_file_address_src,
                _dummy_output_file_address_tgt,
                _tot_num_of_line,
                file_name,
                line_id_dict,
                read_type="w",
                parallel_file=1,
                debug=1):
    _out_file_ptr_src = open(_dummy_output_file_address_src, read_type)
    _out_file_ptr_tgt = open(_dummy_output_file_address_tgt, read_type)
    print("\nReading {0} dataset...".format(file_name))
    tic = time.time()
    cnt = 0
    skipped = 0
    with open(_src_file_address) as src_file_ptr, \
            open(_tgt_file_address) as tgt_file_ptr:
        for src_sent, tgt_sent in zip(src_file_ptr, tgt_file_ptr):
            src_sent = refine(src_sent)
            tgt_sent = refine(tgt_sent)
            if parallel_file:
                if src_sent not in line_id_dict and tgt_sent not in line_id_dict:
                    _out_file_ptr_src.write(src_sent)
                    _out_file_ptr_src.flush()
                    _out_file_ptr_tgt.write(tgt_sent)
                    _out_file_ptr_tgt.flush()
                    cnt += 1
                    if cnt % 10000 == 0 and debug:
                        assert_check_inside(_dummy_output_file_address_src,
                                            _dummy_output_file_address_tgt,
                                            _tot_num_of_line,
                                            cnt)
                        print("Line number :" if cnt <= 10000 else " ", cnt, end="")
                        sys.stdout.flush()
                    line_id_dict.add(src_sent)
                    line_id_dict.add(tgt_sent)
                else:
                    skipped += 1
            else:
                raise Exception("condition for parallel_file == 0 is not implemented in the script")

    assert_check_inside(_dummy_output_file_address_src,
                        _dummy_output_file_address_tgt,
                        _tot_num_of_line,
                        cnt)
    print("Line number :" if cnt <= 10000 else " ", cnt, end="")
    print("")
    toc = time.time()
    total_time = round(toc - tic, 3)
    print("{0} dataset reading time {1}(s)".format(file_name, total_time))
    return line_id_dict, skipped


def create_file_path(dir,
                     dataset_name,
                     dataset_type,
                     src_lang,
                     tgt_lang,
                     suffix,
                     verbose):
    nmt_str = '.' + src_lang + '-' + tgt_lang + '.' + suffix
    _path = os.path.join(dir, dataset_name)
    if verbose == 2:
        print("\n\tFile path creation for", suffix, "language")
        print("\t", "-" * 50)
        print("\t", "current_folder :", dir)
        print("\t", "datase folder :", dataset_name)
        print("\t", "new address :", _path)

    __path = os.path.join(_path, dataset_type)
    if verbose == 2:
        print("\t", "current_folder :", _path)
        print("\t", "dataset_type folder :", dataset_type)
        print("\t", "new address :", __path)

    ___path = os.path.join(__path, dataset_name) + '.' +\
              dataset_type + nmt_str
    if verbose == 2:
        print("\t", "current_folder :", __path)
        print("\t", "final address :", ___path)
        print("\t", "-" * 50)
    return ___path


def create_out_file_address(dataset_types,
                            out_dir,
                            parallel_file,
                            verbose,
                            src_lang,
                            tgt_lang):
    address_dict = {}
    dummy_output_file_name = 'dummy'
    if parallel_file:
        for dataset_type in dataset_types:
            out_dir_dt = os.path.join(out_dir, dataset_type)
            os.makedirs(out_dir_dt, exist_ok=True)
            if verbose == 2:
                print(out_dir_dt, "folder created")
            dummy_output_file_address_src = os.path.join(out_dir_dt, dummy_output_file_name) +\
                                            '.' + dataset_type + '.' + src_lang + '-' + \
                                            tgt_lang + '.' + src_lang
            dummy_output_file_address_tgt = os.path.join(out_dir_dt, dummy_output_file_name) +\
                                            '.' + dataset_type + '.' + src_lang + '-' + \
                                            tgt_lang + '.' + tgt_lang
            address_dict[dataset_type] = (dummy_output_file_address_src, dummy_output_file_address_tgt)
            if verbose == 2:
                print("src", dataset_type, "set address :", dummy_output_file_address_src)
                print("tgt", dataset_type, "set address :", dummy_output_file_address_tgt)
    else:
        raise Exception("condition for parallel_file == 0 is not implemented in the script")

    return address_dict


def read_and_write_test(src_num_of_line,
                        tgt_num_of_line,
                        ref_num_of_line,
                        verbose):
    try:
        if verbose >= 1:
            print("\nREAD and WRITE test: testing if there is any reduction or addition"
                  "for python read and write.")
        assert src_num_of_line == ref_num_of_line == tgt_num_of_line
    except AssertionError:
        print("src_num_of_line (newly read data) :", src_num_of_line)
        print("tgt_num_of_line (newly read data) :", tgt_num_of_line)
        print("ref_num_of_line (taken from data_set_accumulation() ) :".
              format(ref_num_of_line))
        raise
    if verbose >= 1:
        print("Passes the test.\n")
    return 0


def total_num_of_line_test(tot_num_of_line, skipped_dict, read_tot_num_of_line, verbose):
    for dataset_type, cnt in tot_num_of_line.items():
        try:
            if verbose >= 1:
                print("\nTesting is sum of total number of train, dev and test is same.")
            assert read_tot_num_of_line[dataset_type]-skipped_dict[dataset_type] == cnt
        except AssertionError:
            print("dataset type :", dataset_type)
            print("total number of example (newly read):",
                  read_tot_num_of_line[dataset_type])
            print("total number of example (taken from data_set_accumulation() ):",
                  cnt)
            raise
        if verbose >= 1:
            print("Passed the test.\n")
    return 0


def data_set_accumulation(params):

    print("Arg values")
    print(params)
    print("")

    all_files = os.listdir(params.dir)
    all_files.sort(key=lambda x: x.lower())
    dataset_line = {}
    params.restrict = list(map(str, params.restrict.split()))
    (file_dict,
     output_file_name) = retrieve_file_dict(all_files,
                                            params.src_lang,
                                            params.tgt_lang)

    os.makedirs(params.out_dir, exist_ok=True)
    if params.verbose == 2:
        print(params.out_dir, "folder created")

    dataset_types = ['dev', 'test', 'train']
    dataset_types.sort()    # very important to do dev and test at first then train
    print("Accumulating dev, test and then train dataset.")
    address_dict = create_out_file_address(dataset_types,
                                           params.out_dir,
                                           params.parallel_file,
                                           params.verbose,
                                           params.src_lang,
                                           params.tgt_lang)
    
    tot_num_of_line = {}
    skipped_dict = {}
    for dataset_type in dataset_types:
        tot_num_of_line[dataset_type] = 0
        skipped_dict[dataset_type] = 0

    dataset_add_list = []
    line_id_dict = set()
    for dataset_type in dataset_types:
        print("\n\nCREATING `{0}` DATASET ...".format(dataset_type))
        read_type_flag = 0
        for val in file_dict:

            print("\n", "#" * 100, "\n", val, " dataset (", dataset_type, ")", "\n", "#" * 100, sep="")
            if val in params.restrict:
                print("{0} : is manually restricted to read.\n".format(val))
                continue

            # create file path
            # ./params.dir/[alt/ubuntu/os16/...]/[train/dev/test]/[alt/ubuntu/os16/...].en-ms.en
            src_file_address = create_file_path(params.dir,
                                                val,
                                                dataset_type,
                                                params.src_lang,
                                                params.tgt_lang,
                                                params.src_lang,
                                                verbose=params.verbose)
            if params.verbose >= 1:
                print("\t", "src_file_address :", src_file_address)
            if os.path.exists(src_file_address):
                tgt_file_address = create_file_path(params.dir,
                                                    val,
                                                    dataset_type,
                                                    params.src_lang,
                                                    params.tgt_lang,
                                                    params.tgt_lang,
                                                    verbose=params.verbose)

                if params.verbose >= 1:
                    print("\t", "tgt_file_address :", tgt_file_address)

                dataset_add_list.append((src_file_address, tgt_file_address))

                if os.path.exists(tgt_file_address):

                    num_lines_src, num_lines_tgt = assert_check_before(src_file_address,
                                                                       tgt_file_address)
                    file_name = val + '.' + dataset_type
                    print("\n\tTotal number of line in the file {0}: {1}".format(file_name, num_lines_src))

                    # Write on the combined output file
                    read_type = "w" if read_type_flag == 0 else "a"
                    if params.verbose == 1:
                        print("\tLines will be {0} in".format("written" if read_type == "w" else "appended"))
                        print("\t\t", address_dict[dataset_type][0])
                        print("\t\t", address_dict[dataset_type][1])

                    (line_id_dict,
                     skipped) = write_lines(src_file_address,
                                            tgt_file_address,
                                            address_dict[dataset_type][0],
                                            address_dict[dataset_type][1],
                                            tot_num_of_line[dataset_type],
                                            file_name,
                                            line_id_dict,
                                            debug=params.debug,
                                            parallel_file=params.parallel_file,
                                            read_type=read_type)
                    if params.verbose == 1:
                        print("Total {0} number of lines skipped from {1}-{2} dataset".
                              format(skipped, val, dataset_type))
                    read_type_flag = 1
                    # Check if all the lines have been successfully added.
                    assert_check_after(num_lines_src,
                                       num_lines_tgt,
                                       address_dict[dataset_type][0],
                                       address_dict[dataset_type][1],
                                       tot_num_of_line[dataset_type],
                                       parallel_file=params.parallel_file,
                                       skipped_lines=skipped)

                    tot_num_of_line[dataset_type] += num_lines_src-skipped
                    dataset_line[file_name] = num_lines_src
                    skipped_dict[dataset_type] += skipped
                    print("-" * 100)

                else:
                    print("{0} file exists for src language but not for tgt language".
                          format(src_file_address))
            else:
                print("{0} file doesn't contain nmt input file naming convension or doesn't exists.".
                      format(src_file_address))

    return (dataset_types,
            tot_num_of_line,
            skipped_dict,
            dataset_line,
            dataset_add_list,
            output_file_name)


def reporting_read_write_check(params,
                               tot_num_of_line,
                               skipped_dict,
                               dataset_line,
                               dataset_add_list,
                               output_file_name):
    print("\nReporting and read&write check started.")
    print("*" * 100)
    read_tot_num_of_line = {}
    if params.parallel_file:
        raw_tot_line = 0
        for dataset_add in dataset_add_list:
            language_pair = '.' + params.src_lang + "-" + params.tgt_lang
            file_name = os.path.basename(dataset_add[0]).split(language_pair)[0]
            dataset_type = file_name.split(".")[1]
            src_num_of_line = sum(1 for _ in open(dataset_add[0]))
            tgt_num_of_line = sum(1 for _ in open(dataset_add[1]))
            read_and_write_test(src_num_of_line,
                                tgt_num_of_line,
                                dataset_line[file_name],
                                params.verbose)
            print("Total {0} lines in {1}".format(src_num_of_line, file_name))
            if dataset_type not in read_tot_num_of_line:
                read_tot_num_of_line[dataset_type] = src_num_of_line
            else:
                read_tot_num_of_line[dataset_type] += src_num_of_line
            raw_tot_line += src_num_of_line

        for k, v in skipped_dict.items():
            raw_tot_line -= v

        # reading train, dev and test from source and current run's variable.
        total_num_of_line_test(tot_num_of_line, skipped_dict, read_tot_num_of_line, params.verbose)

        read_tot_num_of_line = {}
        raw_tot_line_output = 0

        for dataset_type in tot_num_of_line:
            src_out_address = os.path.join(
                                os.path.join(params.out_dir, dataset_type),
                                'dummy') + '.' + dataset_type + '.' + \
                                params.src_lang + '-' + params.tgt_lang + '.' + \
                                params.src_lang
            tgt_out_address = os.path.join(
                                os.path.join(params.out_dir, dataset_type),
                                'dummy') + '.' + dataset_type + '.' + \
                                params.src_lang + '-' + params.tgt_lang + '.' + \
                                params.tgt_lang
            src_num_of_line = sum(1 for _ in open(src_out_address))
            tgt_num_of_line = sum(1 for _ in open(tgt_out_address))
            try:
                if params.verbose >= 1:
                    print("\nTesting if number of line in paraller output dataset is same or not.")
                assert src_num_of_line == tgt_num_of_line
            except AssertionError:
                print("src_num_of_line :", src_num_of_line)
                print("tgt_num_of_line :", tgt_num_of_line)
                raise
            if params.verbose >= 1:
                print("Test passed.\n")
            read_tot_num_of_line[dataset_type] = src_num_of_line
            raw_tot_line_output += src_num_of_line

        # reading train, dev and test from source and output data variable.
        total_num_of_line_test(tot_num_of_line, {'dev': 0, 'test': 0, 'train': 0}, read_tot_num_of_line, params.verbose)
        try:
            assert raw_tot_line_output == raw_tot_line
        except AssertionError:
            print("raw_tot_line_output:", raw_tot_line_output)
            print("raw_tot_line:", raw_tot_line)
            raise

        # Rename the file and print the report.
        new_folder = os.path.join(params.out_dir, output_file_name)
        os.makedirs(new_folder, exist_ok=True)
        if params.verbose == 2:
            print("new folder created at :", new_folder)

        dataset_final_address = []
        for dataset_type in tot_num_of_line:
            src_out_address = os.path.join(
                                os.path.join(params.out_dir, dataset_type),
                                'dummy') + '.' + dataset_type + '.' + \
                                params.src_lang + '-' + params.tgt_lang + '.' + \
                                params.src_lang
            tgt_out_address = os.path.join(
                                os.path.join(params.out_dir, dataset_type),
                                'dummy') + '.' + dataset_type + '.' + \
                                params.src_lang + '-' + params.tgt_lang + '.' + \
                                params.tgt_lang
            new_src_out_address = os.path.join(new_folder, dataset_type) + \
                                  '.' + params.src_lang + \
                                  '-' + params.tgt_lang + '.' + \
                                  params.src_lang
            new_tgt_out_address = os.path.join(new_folder, dataset_type) + \
                                  '.' + params.src_lang + \
                                  '-' + params.tgt_lang + '.' + \
                                  params.tgt_lang

            src_cmd = "cp " + src_out_address + " " + new_src_out_address
            tgt_cmd = "cp " + tgt_out_address + " " + new_tgt_out_address
            os.system(src_cmd)
            os.system(tgt_cmd)

            id_set = set()
            src_ptr = open(new_src_out_address, "r")
            tgt_ptr = open(new_tgt_out_address, "r")
            for src_line, tgt_line in zip(src_ptr, tgt_ptr):
                id_set.add(src_line)
                id_set.add(tgt_line)
            dataset_final_address.append(((dataset_type, id_set), (new_src_out_address, new_tgt_out_address)))
            no_of_line = sum(1 for _ in open(new_src_out_address))
            print(new_src_out_address)
            print(new_tgt_out_address)
            print("Total number of line in {0} : {1}".format(dataset_type, no_of_line))

        print("Calculating number of common lines between dev, test and train.")
        for itr1, ((dataset_type1, id_set1), (src_add1, tgt_add1)) in enumerate(dataset_final_address):
            for itr2, ((dataset_type2, id_set2), (_, _)) in enumerate(dataset_final_address):
                if itr1 == itr2:
                    continue
                src_add1_ptr = open(src_add1, "r")
                tgt_add1_ptr = open(tgt_add1, "r")
                cnt = 0
                for src_add1_line, tgt_add1_line in zip(src_add1_ptr, tgt_add1_ptr):
                    if src_add1_line in id_set2 or tgt_add1_line in id_set2:
                        cnt += 1
                print("Total {0} number of line common between {1} and {2}.".
                      format(cnt, dataset_type1, dataset_type2))

#################################################
# Start of the script
# TODO: implement condition for parallel_file==0
#################################################

def main():
    params = parser.parse_args()
    (dataset_types,
     tot_num_of_line,
     skipped_dict,
     dataset_line,
     dataset_add_list,
     output_file_name) = data_set_accumulation(params)
    reporting_read_write_check(params,
                               tot_num_of_line,
                               skipped_dict,
                               dataset_line,
                               dataset_add_list,
                               output_file_name)


if __name__ == "__main__":
    main()
