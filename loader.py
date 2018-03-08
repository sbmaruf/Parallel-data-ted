import os
import numpy as np 


debug = 1
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def isOk(words):
    if words[1] == '-->' \
            and len(words[0].split(':')) == 3 \
            and len(words[2].split(':')) == 3 \
            and len(words[0].split(':')[2].split(',')) == 2 \
            and len(words[2].split(':')[2].split(',')) == 2:
        return True
    return False

def extract_line(File, file_name):
    state = 0 
    line_no = 1
    dict={}
    idx = 0
    tmp_line = ''
    for line in File:
        words = line.strip().split()
        n_w = len(words)
        if state == 2 :
            tmp_line = tmp_line + line
        if n_w == 3 and isOk(words) and state == 1:
            state = state+1
            idx = words[0]+"+"+words[2]
        if n_w == 0 :
            if state == 2 :
                state = state + 1
                dict[idx] = tmp_line
                tmp_line = ''
                line_no = line_no+1
        if n_w == 1 and is_number(words[0]) and state == 0:
            num = int(words[0])
            if num != line_no :
                if debug == 1 :
                    print("Error in", file_name, ".\nAt text file line is:", num , ", in code line is:",line_no )
                assert( num == line_no )
            state = state + 1
        state = state % 3
    if idx not in dict:
        dict[idx] = tmp_line
    if len(dict) != line_no :
        if debug :
            print("File name :", file_name)
            print("Total line in the Dictionary :",len(dict))
            print("Toal line retrieved :", line_no) 
        assert(len(dict) == line_no)
    
    return dict   


lang1 = "English"
lang2 = "Malay"
directory = "./Ma-En/"
objects = os.listdir(directory)
objects.sort()
cnt = tot_file = 0 
err_cnt = np.zeros(10000)
for file_name in objects:
    # print(os.path.splitext(file_name))
    file_ext = os.path.splitext(file_name)
    name = file_ext[0]
    ext = file_ext[1]
    if name.split('.')[0] != lang1 :
        continue
    
    if os.path.splitext(file_name)[1].lower() == ".srt":
        video_hash = name.split('.')[1]
        dict_time_stamp_lang1={}
        dict_time_stamp_lang2={}
        print("Opening ", file_name, " and ", lang2+"."+video_hash+ext , "\n" )
        with open(directory+file_name) as lang1_file, open(directory+lang2+"."+video_hash+ext) as lang2_file :
            dict_time_stamp_lang1 = extract_line(lang1_file, file_name)
            dict_time_stamp_lang2 = extract_line(lang2_file, lang2+"."+video_hash+ext)
            if abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))>159:
                print("Warning : Number of line is not same for", file_name, " and ", lang2+"."+video_hash+ext)
                print(len(dict_time_stamp_lang1),len(dict_time_stamp_lang2))
                input("Enter to continue")                   
                err_cnt[abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))]=err_cnt[abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))]+1
                cnt = cnt + 1 
        tot_file = tot_file + 1

if debug :
    print("Total number of Warning", cnt)
    print("Total number of file", tot_file)
    tot = 0 
    print("difference  --> # of example")
    print("----------------------------")
    for idx,val in enumerate(err_cnt):
        if val > 0:
            print("      ", idx,"    -->    ", val)
        tot = tot + val
    assert( tot == cnt )
# print(cnt) 


