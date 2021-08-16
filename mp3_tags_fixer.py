# -*- coding: utf-8 -*-

import os
import eyed3
import sys
# from multiprocessing import Pool


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def chek_field_is_non_or_english(s):
    if not s is None:
        return is_english(s)
    return True

def decode_russian(s):
    try:
        print (s)
        return s.encode('cp1252').decode('cp1251')
    except UnicodeEncodeError:
        return s.encode().decode()
    except UnicodeDecodeError:
        return ""

def add_russion_files_to_list(el):
    # count +=1
    # print (el)
    audiofile = eyed3.load(el)
    # print (count, el)
    if audiofile and audiofile.tag \
        and (not chek_field_is_non_or_english(audiofile.tag.artist) \
        or not chek_field_is_non_or_english(audiofile.tag.album) \
        or not chek_field_is_non_or_english(audiofile.tag.title) ):
            return (el)
    else:
            return ""

def make_russian_good(el):
    print(el)
    audiofile = eyed3.load(el)
    if not chek_field_is_non_or_english(audiofile.tag.artist):
        audiofile.tag.artist = decode_russian(audiofile.tag.artist)
    if not chek_field_is_non_or_english(audiofile.tag.album):
        audiofile.tag.album = decode_russian(audiofile.tag.album)
    if not chek_field_is_non_or_english(audiofile.tag.title):
        audiofile.tag.title = decode_russian(audiofile.tag.title)
    audiofile.tag.save(version=eyed3.id3.ID3_DEFAULT_VERSION,encoding='utf-8')
    

def main():
    
    dd = []
    for roots, dirs, files in os.walk(sys.argv[1]):
        for name in files:
            if name.endswith(".mp3"):
                dd.append(os.path.abspath(os.path.join(roots, name)))

    
    # print (dd[1])
    # dd = filter (lambda x: x.find("_rap") == -1, dd)

    count = 0
    ddd1 = []

    for el in dd:
        if add_russion_files_to_list(el):
            ddd1.append(el)

    # with Pool(16) as pool:
    #     pool.map(add_russion_files_to_list, dd)

    # add_russion_files_to_list(dd, ddd1)

    # print ("found ", len(dd), "mp3 tracks")
    print ("non english ", len(ddd1))
    # print (".\n".join(ddd1))

    for el in ddd1:
        make_russian_good(el)


    # kk.encode('cp1252').decode('cp1251')


if __name__ == "__main__":
    main()
