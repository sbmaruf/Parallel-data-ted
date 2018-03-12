import os
import numpy as np 
import argparse
import shutil

parser = argparse.ArgumentParser(description='Ted parallel data extraction.')
parser.add_argument("--debug", type=int, default=1, help="Additional print operation for debug.")
parser.add_argument("--drop_dir", type=str, default='./error_data/', help="The dataset with ambiguous data will be transferred to this directory.")
parser.add_argument("--per_dir", type=str, default='./perfect_data/', help="The dataset with perfect data will be transferred to this directory.")
parser.add_argument("--war_dir", type=str, default='./warning_data/', help="The dataset with warning based data will be transferred to this directory.")
parser.add_argument("--lang1",type=str, default="English", help="Each of the srt file's name starts with this Language name.")
parser.add_argument("--lang2",type=str, default="Malay", help="Each of the srt file's name starts with this Language name.")
parser.add_argument("--dir", type=str, default="./Ma-En/", help="Directory of the dataset")
parser.add_argument("--out_dir", type=str, default="./nmt_io/", help="NMT parallel sentences files")
parser.add_argument("--out_file", type=str, default="par_sent.txt", help="Name of the output file")
 
params = parser.parse_args()

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


def write_sent(dict1 , dict2, hash_name):
    if not os.path.isdir(params.out_dir):
        os.makedirs(params.out_dir)
    cnt = 0 
    if write_sent.itr == 0:
        perm = "w"
    else:
        perm = "a"
    with open(params.out_dir+params.out_file, perm) as f: 
        for i,v in dict1.items():
            #print(i,v)
            if i in dict2:
                #print("----------------->",i,dict2[i])
                f.write("#-----<s>-----#\n")
                f.write(dict1[i]+"\n")
                f.write("#-----<s>-----#\n")
                f.write(dict2[i]+"\n")
                cnt += 1
    if params.debug == 1 :
        print("----------\nfile name:",hash_name)
        print("tot number of line(lang1-lang2):",len(dict1),len(dict2))
        print("number of line extracted:",cnt)
        print("----------")
        #input("are you ok to continue")
    write_sent.itr = 1
    return cnt

write_sent.itr = 0


def extract_line(File, file_name):
    state = 0 
    line_no = flag = 1
    dict={}
    idx = 0
    tmp_line = ''
    for line in File:
        words = line.strip().split()
        n_w = len(words)
        if state == 2 :
            tmp_line = tmp_line + ' ' + line.strip()
        #print(line.strip(), state, n_w, tmp_line, idx)
        if n_w == 3 and isOk(words) and state == 1:
            state = state+1
            idx = words[0].strip()+"+"+words[2].strip()
        if n_w == 0 :
            if state == 2 :
                state = state + 1
                dict[idx] = tmp_line
                tmp_line = ''
                line_no = line_no+1
        if n_w == 1 and is_number(words[0]) and state == 0:
            num = int(words[0])
            if num != line_no :
                if params.debug == 1 :
                    print("Error in", file_name, ".\nAt text file line is:", num , ", in code line is:",line_no )
                assert( num == line_no )
            state = state + 1
        state = state % 3
    if idx not in dict:
        dict[idx] = tmp_line
        line_no = line_no+1
    if len(dict)+1 != line_no :
        if params.debug :
            print("File name :", file_name)
            print("Total line in the Dictionary :",len(dict))
            print("Toal line retrieved :", line_no) 
        # assert(len(dict) == line_no)
        flag = 0 
    
    return dict, flag  


objects = os.listdir(params.dir)
objects.sort()

tot_file = critical = perfect = warning = 0 
tot_line_lang1_war = tot_line_lang2_war = 0
tot_line_lang1_per = tot_line_lang2_per = 0
tot_line_lang1_war_extracted = tot_line_lang2_war_extracted = 0
tot_line_lang1_per_extracted = tot_line_lang2_per_extracted = 0

err_cnt = np.zeros(10000)
for file_name in objects:
    # print(os.path.splitext(file_name))
    file_ext = os.path.splitext(file_name)
    name = file_ext[0]
    ext = file_ext[1]
    if name.split('.')[0] != params.lang1 :
        continue
    
    if os.path.splitext(file_name)[1].lower() == ".srt":
        video_hash = name.split('.')[1]
        dict_time_stamp_lang1={}
        dict_time_stamp_lang2={}
        print("Opening ", file_name, " and ", params.lang2+"."+video_hash+ext)
        with open(params.dir+file_name) as lang1_file, open(params.dir+params.lang2+"."+video_hash+ext) as lang2_file :
            
            dict_time_stamp_lang1, flag1 = extract_line(lang1_file, file_name)
            dict_time_stamp_lang2, flag2 = extract_line(lang2_file, params.lang2+"."+video_hash+ext)
            
            if abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))>0:
                print("Warning : Number of line is not same for", file_name, "("+str(len(dict_time_stamp_lang1))+")",  " and ", params.lang2+"."+video_hash+ext, "("+str(len(dict_time_stamp_lang2))+")")
                err_cnt[abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))]=err_cnt[abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))]+1
                warning = warning + 1
                if not os.path.isdir(params.war_dir):
                    os.makedirs(params.war_dir)
                if params.debug == 1 :
                    print("Item copied from",params.dir+file_name, "to", params.drop_dir+file_name)
                    print("Item copied from",params.dir+params.lang2+"."+video_hash+ext, "to",  params.drop_dir+params.lang2+"."+video_hash+ext)
                shutil.copy(params.dir+file_name, params.war_dir+file_name)
                shutil.copy(params.dir+params.lang2+"."+video_hash+ext, params.war_dir+params.lang2+"."+video_hash+ext)
                tot_ext = write_sent(dict_time_stamp_lang1,dict_time_stamp_lang2, video_hash)
                tot_line_lang1_war += len(dict_time_stamp_lang1)
                tot_line_lang2_war += len(dict_time_stamp_lang2)
                tot_line_lang1_war_extracted += tot_ext
                tot_line_lang2_war_extracted += tot_ext

            elif abs(len(dict_time_stamp_lang1)-len(dict_time_stamp_lang2))==0 and flag1 == True and flag2 == True :
                 if not os.path.isdir(params.per_dir):
                    os.makedirs(params.per_dir)
                 if params.debug == 1 :
                     print("Item copied from",params.dir+file_name, "to", params.per_dir+file_name)
                     print("Item copied from",params.dir+params.lang2+"."+video_hash+ext, "to",  params.per_dir+params.lang2+"."+video_hash+ext)
                 shutil.copy(params.dir+file_name, params.per_dir+file_name)
                 shutil.copy(params.dir+params.lang2+"."+video_hash+ext, params.per_dir+params.lang2+"."+video_hash+ext)
                 perfect = perfect + 1 
                 tot_ext = write_sent(dict_time_stamp_lang1,dict_time_stamp_lang2, video_hash)
                 tot_line_lang1_per += len(dict_time_stamp_lang1)
                 tot_line_lang2_per += len(dict_time_stamp_lang2)
                 tot_line_lang1_per_extracted += tot_ext
                 tot_line_lang2_per_extracted += tot_ext

            if flag1 == False or flag2 == False :
                critical = critical + 1 
                if not os.path.isdir(params.drop_dir):
                    os.makedirs(params.drop_dir)
                if params.debug == 1 :
                    print("Item copied from",params.dir+file_name, "to", params.drop_dir+file_name)
                    print("Item copied from",params.dir+params.lang2+"."+video_hash+ext, params.drop_dir+params.lang2+"."+video_hash+ext)
                shutil.copy(params.dir+file_name, params.drop_dir+file_name)
                shutil.copy(params.dir+params.lang2+"."+video_hash+ext, params.drop_dir+params.lang2+"."+video_hash+ext)

        tot_file = tot_file + 1

if params.debug :
    print("\n\n\nGenerating reports")
    print("--------------------")
    print("Total number of file:", tot_file*2)
    print("Total number of Warned file:", warning*2)
    print("Total number of errored file:",critical*2)
    print("Total number of perfect file:",perfect*2)
    tot = 0 
    print("difference  --> # of example")
    print("----------------------------")
    for idx,val in enumerate(err_cnt):
        if val > 0:
            print("      ", idx,"    -->    ", int(val))
        tot = tot + val
    assert( tot == warning )
    print("Total number of line in language 1 from warned file:", tot_line_lang1_war)
    print("Total number of line in language 2 from warned file:", tot_line_lang2_war)
    print("Total number of line in language 1 from perfect file:", tot_line_lang1_per)
    print("Total number of line in language 2 from perfect file:", tot_line_lang2_per)
    print("Total pair of line extracted from warned files:", tot_line_lang1_war_extracted)
    print("Total pair of line extracted from perfect files:", tot_line_lang1_per_extracted)
    print("Total pair of line:",tot_line_lang1_war_extracted+tot_line_lang1_per_extracted)
    print("Matching number of line from the generated file contains parallel lines")
    num_lines = sum(1 for line in open(params.out_dir+params.out_file))
    assert(int(num_lines/2)==(tot_line_lang1_war_extracted+tot_line_lang1_per_extracted)*2 )
    print("Lines generated successful!")

