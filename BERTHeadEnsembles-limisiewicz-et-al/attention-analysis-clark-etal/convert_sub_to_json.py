import utils
import sys

def convert_sub(json_file_path):
    json_with_sub = utils.load_json(json_file_path)
    final = []
    for d in json_with_sub:
        temp_list = [{'text' : d['text']}]
        for pos, sent_list in d['substitutions']:
            if len(sent_list) != 0:
                for s in sent_list:
                    temp_list.append({'text' : s})
        final += temp_list
    return final


if __name__=="__main__":
    PATH_TO_FILE = str(sys.argv[1])
    fully_listed = convert_sub(PATH_TO_FILE)
    outfile=PATH_TO_FILE.split('.')[0] + '_listed.json'
    utils.write_json(fully_listed, outfile)
