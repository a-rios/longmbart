#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# list of commands to run for converting src/tgt files with
# hardcoded mbart-style tags, e.g. </s> de_DE to
# longmbart-style prepended lang tags, e.g. de_DE src-sequence.

grep -Po '\S+$' test.review > test.lang_tags
grep -Po '\S+$' valid.review > valid.lang_tags
grep -Po '\S+$' train.review > train.lang_tags

sed -i -E "s/ <\/s> \S+$//" test.review
sed -i -E "s/ <\/s> \S+$//" valid.review
sed -i -E "s/ <\/s> \S+$//" train.review

sed -i -E "s/ <\/s> \S+$//" test.response
sed -i -E "s/ <\/s> \S+$//" valid.response
sed -i -E "s/ <\/s> \S+$//" train.response

sed -i -E "s/ <\/s> \S+$//" test.review_dom_est_rat
sed -i -E "s/ <\/s> \S+$//" valid.review_dom_est_rat
sed -i -E "s/ <\/s> \S+$//" train.review_dom_est_rat

paste -d' ' test.lang_tags test.review > test.review_tagged
paste -d' ' valid.lang_tags valid.review > valid.review_tagged
paste -d' ' train.lang_tags train.review > train.review_tagged

paste -d' ' test.lang_tags test.response > test.response_tagged
paste -d' ' valid.lang_tags valid.response > valid.response_tagged
paste -d' ' train.lang_tags train.response > train.response_tagged

paste -d' ' test.lang_tags test.review_dom_est_rat > test.review_tagged_dom_est_rat
paste -d' ' valid.lang_tags valid.review_dom_est_rat > valid.review_tagged_dom_est_rat
paste -d' ' train.lang_tags train.review_dom_est_rat > train.review_tagged_dom_est_rat


