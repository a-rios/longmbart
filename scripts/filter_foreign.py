#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
import sys


filters = [
    # chinese
    re.compile(r"[\u4e00-\u9fff|\u31c0-\u31ef|\u31f0-\u31ff|\u3200-\u32ff|\u3300-\u33ff|\u3400-\u4dbf|\u4dc0-\u4dff|\u4e00-\u9fff|\uf900-\ufaff|\u2e80-\u2eff|\u2f00-\u2fdf|\u3000-\u303f|\u3100-\u312f|\u3190-\u319f|\u31a0-\u31bf|\ufe30-\ufe4f]+"),
    
    # hangul
    re.compile(r"[\u1100-\u1112|\u1161-\u1175|\u11a8-\u11c2|\uac00-\ud7af|\u3130-\u318f]+"),
    
    # ethiopic
    re.compile(r"[\u1200-\u137f]+"),
    
    # arabic
    re.compile(r"[\u0600-\u06ff|\ufb50-\ufdff|\ufe70-\ufeff|\u0700-\u074f]+"),
    
    # devanagari
    re.compile(r"[\u0900-\u097f|\u0b80-\u0bff|\u0b00-\u0b7f|\u0a00-\u0a7f|\u0980-\u09ff|\u0a80-\u0aff|\u0c80-\u0cff|\u0c00-\u0c7f|\u0d00-\u0d7f|\u0d80-\u0dff]+"),
    
    # canadian
    re.compile(r"[\u1400-\u167f]+"),
    
    # japanese
    re.compile(r"[\u4e00-\u9fbf|\u3040-\u309f|\u30a0-\u30ff|\u31F0-\u31FF]+"),
    
    # myanmar
    re.compile(r"[\u1000-\u109f]+"),
    
    # thai
    re.compile(r"[\u0e00-\u0e7f|\u0e80-\u0eff]+"), # + lao[\u0e80-\u0eff]
    
    # tibetan
    re.compile(r"[\u0f00-\u0fff]+"),
    
    # phags pa
    re.compile(r"[\ua840-\ua87f]+"),
    
    # mongolian
    re.compile(r"[\u1800-\u18aa]+"),
    
    # yi
    re.compile(r"[\ua000-\ua48f|\ua490-\ua4cf]+"), # not contained
    
    # cham
    re.compile(r"[\uaa00-\uaa5f]+"),
    
    # georgian
    re.compile(r"[\u10a0-\u10ff]+"),
    
    # khmer
    re.compile(r"[\u1780-\u17ff]+"),
    
    # cyrillic
    re.compile(r"[\u0400-\u04ff][\u0400-\u04ff]+"), ## leave single letters, but remove larger pieces
    
    # greek
    re.compile(r"[\u0370-\u03ff][\u0370-\u03ff]+"),
    
    # hebrew
    re.compile(r"[\u0590-\u05ff]+"),
    
    # turkic languages
    re.compile(r"[\u011e|\u011f|\u0130|\u0131|\u015e|\u015f]+"),
    
    # general punctuation
    re.compile(r"[\u2000-\u206f]+"),
    
    # ascii punctuation (leave single characters, but remove sequences of identical characters)
    re.compile(r"([\u0020-\u002f|\u003a-\u0040|\u005b-\u0060|\u007b-\u007e|\u00a1-\u00bf])\1+"),
    
    # non-german diacritics/letters (all diacritics/letters except äöüéèàëß in uppercase and lowercase)
    # allowed: \u00c0, \u00c4, \u00c8, \u00c9, \u00cb, \u00d6, \u00dc, \u00df, \u00eo, \u00e4, \u00e8, \u00e9, \u00f6, \u00fc
    # re.compile(r"[\u00c1-\u00c3|\u00c5-\u00c7|\u00ca|\u00cc-\u00d5|\u00d8-\u00db|\u00dd-\u00de|\u00e1-\u00e3|\u00e5-\u00e7|\u00ea-\u00f5|\u00f8-\u00fb|\u00fd-\u00ff]"),
    
    # emojis and symbols
    re.compile(r"[\U0001f600-\U0001f6a9|\U0001f900-\U0001f9ff]+"), #emojis
    re.compile(r"[\u2600-\u26ff]+"), # misc symbols
    re.compile(r"[\U0001f300-\U0001f5ff]+"), # misc symbols and pictographs
    re.compile(r"[\U0001f680-\U0001f6ff]+"), # transport and map symbols
    re.compile(r"[\U0001f100-\U0001f1ff]+"), # enclosed alphanumeric supplement
    re.compile(r"[\u2700-\u27bf]+"), # dingbats
    re.compile(r"[\u2300-\u23ff]+"), # misc technical
    # re.compile(r"[]+"), #
]

class UnwantedSeq(Exception):
    pass


def filter_foreign_characters(unfiltered, return_set=False):
    filtered = list()

    for line in unfiltered:
        try:
            line = line.rstrip()
            # line = re.sub(r"[\u2580-\u259f]+", '', line)
            if len(line) == 0: raise UnwantedSeq
            for f in filters:
                ex = f.findall(line)
                if len(ex) > 0: raise UnwantedSeq
            if re.search(r'\d', line):
                if re.search(r'^\d$|^\d\d\d\d$', line.rstrip()):
                    pass
                else:
                    raise UnwantedSeq
            filtered.append(line)
        except UnwantedSeq:
            continue
            
    filtered = set(filtered)
    if not return_set:
        filtered = sorted(filtered)
    return filtered

                                                                  
if __name__ == "__main__":
    filtered = filter_foreign_characters(sys.stdin.readlines())
    for line in filtered:
        print(line)