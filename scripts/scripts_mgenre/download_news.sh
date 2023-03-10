#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# for LANG in ar bg bs ca cs de el en eo es fa fi fr he hu it ja ko nl no pl pt ro ru sd sq sr sv ta th tr uk zh

mkdir wikinews
cd wikinews

for LANG in en fr
    wget http://wikipedia.c3sl.ufpr.br/${LANG}wikinews/20230220/${LANG}wikinews-20230220-pages-articles-multistream.xml.bz2
done

for LANG in en fr
do
    wikiextractor ${LANG}wikinews-20230220-pages-articles-multistream.xml.bz2 -o ${LANG} --links --section_hierarchy --lists --sections
done

#if [[ ! -f $wikidata_json_dump ]]; then
#  echo "downloading $wikidata_json_dump"
#  wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 -O $wikidata_json_dump
#fi