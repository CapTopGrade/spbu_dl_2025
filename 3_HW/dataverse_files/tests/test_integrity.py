import sys
import zipfile

import unittest
from pandas import read_csv, isnull
from os import walk
from os.path import join, basename
from re import findall, fullmatch, search


class DataIntegrityTestCase(unittest.TestCase):
    METADATA = "metadata.tsv"
    TEXTS = "texts.zip"
    OEUVRE = "https://dataverse.pushdom.ru/api/access/datafile/5747"

    def setUp(self):
        self.terminus_post_quem = list(range(1799, 1837 + 1))
        self.terminus_ante_quem = list(range(1799, 1837 + 1))
        self.lifetime_publication_values = ["да", "нет"]
        self.metadata = read_csv(filepath_or_buffer=self.METADATA, sep="\t")
        self.texts_filenames = list()
        with zipfile.ZipFile(self.TEXTS) as zfile:
            self.texts_filenames = list(map(lambda f: f.filename, zfile.infolist()))
        self.oeuvre = read_csv(filepath_or_buffer=self.OEUVRE, sep="\t")

    def test_texts_filenames(self):
        """Tests if set of filenames in 'texts' directory is equal to set of values in 'filename' column
        'metadata.tsv' table"""
        query = self.metadata.filename.isna() == False
        filename_column_values = self.metadata[query]["filename"].unique()
        self.assertSetEqual(set1=set(filename_column_values), set2=set(self.texts_filenames),
                            msg="Filenames in 'texts' folder are not equal to values in 'filename' column")

    def test_beginnings_lines(self):
        """Tests if lines of poems do not begin with lowercase letters (except for A573 and A652)"""
        for index, text_filename in enumerate(self.texts_filenames):
            if basename(text_filename) not in ["A573.txt", "A652.txt"]:
                with zipfile.ZipFile(self.TEXTS) as zfile:
                    with zfile.open(text_filename) as file:
                        poem = file.readlines()
                        poem = [line.decode() for line in poem]
                        with self.subTest(index=index, text_filename=text_filename):
                            for line in poem:
                                self.assertTrue(
                                    expr=(not findall(pattern=r"^[a-zа-я]", string=line)),
                                    msg=f"""poem {text_filename}: line {(poem.index(line) + 1)} begins with lowercase
                                    letter""")

    def test_trailing_spaces_lines(self):
        """Tests if lines of poems do not contain trailing spaces"""
        for index, text_filename in enumerate(self.texts_filenames):
            with zipfile.ZipFile(self.TEXTS) as zfile:
                with zfile.open(text_filename) as file:
                    poem = file.readlines()
                    poem = [line.decode() for line in poem]
                    for line in poem:
                        with self.subTest(index=index, text_filename=text_filename, line=(poem.index(line) + 1)):
                            self.assertTrue(
                                expr=(not findall(pattern=r" +$", string=line)),
                                msg=f"""poem {text_filename}: line {(poem.index(line) + 1)} contains trailing
                                spaces""")

    def test_last_lines(self):
        """Tests if last lines of poems are either not empty or contain only space(s)"""
        for index, text_filename in enumerate(self.texts_filenames):
            with zipfile.ZipFile(self.TEXTS) as zfile:
                with zfile.open(text_filename) as file:
                    poem = file.readlines()
                    poem = [line.decode() for line in poem]
                    with self.subTest(index=index, text_filename=text_filename):
                        self.assertTrue(expr=(not fullmatch(pattern=r"\s+", string=poem[-1])),
                                        msg=f"""poem {text_filename}: last line {(poem.index(poem[-1]) + 1)} is empty"""
                                        )
                        self.assertFalse(expr=poem[-1] == "",
                                         msg=f"""poem {text_filename}: last line {(poem.index(poem[-1]) + 1)} is
                                         empty""")

    def test_cross_corpus_index(self):
        """Tests if set of uid's in 'metadata.tsv' table is equal to subset of uid's in 'oeuvre.tsv' table"""
        query = self.oeuvre.uid.isin(self.metadata.uid)
        filtered_oeuvre = self.oeuvre[query]
        self.assertSetEqual(set1=set(filtered_oeuvre.uid), set2=set(self.metadata.uid),
                            msg="uid's in verses' metadata are not identical to uid's in index of Pushkin's works")

    def test_tags(self):
        """Tests if there are equal number of opening and closing tags"""
        pattern_opening_tags = [r"<date>", r"<note>", r"<source>", r"<place>", r"<epigraph>", r"<head>", r"<speaker>",
                                r"<stage>"]
        pattern_closing_tags = [r"</date>", r"</note>", r"</source>", r"</place>", r"</epigraph>", r"</head>",
                                r"</speaker>", r"<stage>"]
        for index, text_filename in enumerate(self.texts_filenames):
            with zipfile.ZipFile(self.TEXTS) as zfile:
                with zfile.open(text_filename) as file:
                    poem = file.readlines()
                    poem = [line.decode() for line in poem]
                    poem = "".join(poem)
                    for i in range(len(pattern_opening_tags)):
                        with self.subTest(index=index, text_filename=text_filename):
                            self.assertTrue(expr=len(findall(pattern=pattern_closing_tags[i], string=poem)) == len(
                                findall(pattern=pattern_opening_tags[i], string=poem)),
                                            msg=f"""poem {text_filename}: tags are not balanced, closing tags {
                                            pattern_closing_tags[i]}: {len(findall(pattern=pattern_closing_tags[i],
                                                                                   string=poem))}, opening tags {
                                            pattern_opening_tags[i]}: {len(findall(pattern=pattern_opening_tags[i],
                                                                                   string=poem))}""")

    def test_language_tag(self):
        """Tests if there are either only latin or cyrillic characters in tag's name"""
        pattern_different_languages_tag = r"</?[a-zA-Z]+[^<>]+[а-яёЁйЙА-Я]+>|</?[а-яёЁйЙА-Я]+[^<>]+[a-zA-Z]+>"
        for index, text_filename in enumerate(self.texts_filenames):
            with zipfile.ZipFile(self.TEXTS) as zfile:
                with zfile.open(text_filename) as file:
                    poem = file.readlines()
                    poem = [line.decode() for line in poem]
                    for line in poem:
                        self.assertTrue(expr=(not findall(pattern=pattern_different_languages_tag, string=line)),
                                        msg=f"""poem {text_filename}: line {(poem.index(poem[-1]) + 1)} contains tag
                                        named with latin and cyrillic characters""")

    def test_terminus_quem(self):
        """Tests if values in "terminus_ante_quem" and "terminus_post_quem" column of "metadata" table are in specified
        range"""
        post = list(self.metadata["terminus_post_quem"])
        ante = list(self.metadata["terminus_ante_quem"])
        for index, (p, a) in enumerate(zip(post, ante)):
            with self.subTest(post=p, ante=a, index=index):
                if not isnull(p):
                    self.assertIn(int(p), self.terminus_post_quem,
                                  f"row {index}: terminus_post_quem {p} is not in the list {self.terminus_post_quem}")
                    self.assertIn(int(a), self.terminus_ante_quem,
                                  f"row {index}: terminus_ante_quem {a} is not in the list {self.terminus_ante_quem}")

    def test_nbsp(self):
        """Tests if there are '\xa0' strings in texts of poems"""
        for index, text_filename in enumerate(self.texts_filenames):
            with zipfile.ZipFile(self.TEXTS) as zfile:
                with zfile.open(text_filename) as text_file:
                    poem = text_file.read().decode()
                    self.assertFalse("\xa0" in poem, f"poem {text_filename} contains non-breaking spaces")

    def test_redundant_spaces(self):
        """Tests if there are redundant spaces in values of text columns in 'metadata.tsv' table"""
        for column in self.metadata.dtypes[self.metadata.dtypes == "object"].index.tolist():
            values = list(self.metadata[column])
            for index, v in enumerate(values):
                with self.subTest(value=v, index=index):
                    if not isnull(v):
                        self.assertFalse(search(pattern="\s{2,}", string=v),
                                         f"column {column}: value {v} contains redundant spaces")

    def test_leading_spaces(self):
        """Tests if there are leading spaces in values of text columns in 'metadata.tsv' table"""
        for column in self.metadata.dtypes[self.metadata.dtypes == "object"].index.tolist():
            values = list(self.metadata[column])
            for index, v in enumerate(values):
                with self.subTest(value=v, index=index):
                    if not isnull(v):
                        self.assertFalse(search(pattern="^\s", string=v),
                                         f"column {column}: value {v} contains leading spaces")

    def test_trailing_spaces(self):
        """Tests if there are trailing spaces in values of text columns in 'metadata.tsv' table"""
        for column in self.metadata.dtypes[self.metadata.dtypes == "object"].index.tolist():
            values = list(self.metadata[column])
            for index, v in enumerate(values):
                with self.subTest(value=v, index=index):
                    if not isnull(v):
                        self.assertFalse(search(pattern="\s$", string=v),
                                         f"column {column}: value {v} contains trailing spaces")

    def test_lifetime_publication(self):
        """Tests if there are values in 'lifetime_publication' column of 'metadata.tsv' table that are not equal to
        either 'да' or 'нет'"""
        lifetime_publication = list(self.metadata["lifetime_publication"])
        for index, v in enumerate(lifetime_publication):
            with self.subTest(value=v, index=index):
                if not isnull(v):
                    self.assertIn(v, self.lifetime_publication_values,
                                  f"""index {index}: value {v} is not equal to one of possible 'lifetime_publication'
                                  values""")

    def test_link_uid(self):
        """Tests if there are values in 'link_uid' column of 'metadata.tsv' table that are not contained in 'link_uid'
        column of 'metadata.tsv' table"""
        link_uid_s_metadata = set(self.metadata["link_uid"])
        link_uid_s_oeuvre = set(self.oeuvre["link_uid"])
        self.assertTrue(link_uid_s_metadata.issubset(link_uid_s_oeuvre),
                        f"{link_uid_s_metadata - link_uid_s_oeuvre} in 'metadata.tsv' are not in 'oeuvre.tsv'")

    def test_volumes(self):
        """Tests if there are values in 'volume' column of 'metadata.tsv' table that are not contained in 'volume'
        column of 'metadata.tsv' table"""
        volumes_metadata = set(self.metadata["volume"])
        volumes_oeuvre = set(self.oeuvre["volume"])
        self.assertTrue(volumes_metadata.issubset(volumes_oeuvre),
                        f"{volumes_metadata - volumes_oeuvre} in 'metadata.tsv' that is not in 'oeuvre.tsv'")

    def test_n_lines(self):
        """Tests if there are values in 'n_lines' column of 'metadata.tsv' table that are not NA and less than zero"""
        n_lines = list(self.metadata["n_lines"])
        for index, v in enumerate(n_lines):
            with self.subTest(value=v, index=index):
                if not isnull(v):
                    self.assertTrue(v >= 0, f"index {index}: value {v} is less than zero")

    def test_n_words(self):
        """Tests if there are values in 'n_words' column of 'metadata.tsv' table that are not NA and less than zero"""
        n_words = list(self.metadata["n_words"])
        for index, v in enumerate(n_words):
            with self.subTest(value=v, index=index):
                if not isnull(v):
                    self.assertTrue(v >= 0, f"index {index}: value {v} is less than zero")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DataIntegrityTestCase.METADATA = sys.argv.pop()
    unittest.main()
