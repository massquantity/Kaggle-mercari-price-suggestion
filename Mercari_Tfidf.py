import numpy as np
import pandas as pd
import nltk
import re
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from scipy.sparse import csc_matrix, csr_matrix, hstack
from lightning.regression import AdaGradRegressor, CDRegressor, SDCARegressor
import time
import sys
import warnings
warnings.filterwarnings('ignore')


def size_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb


def sparse_mb(sparse_df):
    size_in_bytes = sparse_df.data.nbytes
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb


print('START')
start_time = time.time()
train = pd.read_csv("../input/train.tsv", sep='\t')
org = train.shape[0]
train.drop(train[train.price < 1.0].index, axis=0, inplace=True)
print('reduced lines: ', org - train.shape[0])
print("load_time: ", time.time() - start_time)
print("Train size: {:.2f} mb".format(size_mb(train)))

y = np.array(np.log1p(train.price))

pattern = r'\bvictoria secret\b|\bps[1-4]\b|(?:iphone)(?:7plus|6s?plus|3|4s?|5s?|6s?|7s?|se)|\d+\.\d+%|\d+%|' \
          r'(?<=\s)\.\d+|\d+\.\d+|\d+|[a-zA-Z]+|(?<=\d)\$'
pattern111 = r'\w+'


class cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X["brand_name"].fillna("missbrand", inplace=True)
        X["brand_name"] = X.brand_name.str.lower()
        X["brand_name"] = X.brand_name.str.replace(r"®|‘|’|™", "")
        X["brand_name"] = X.brand_name.str.replace(r"victoria\'s secret", "victoria secret")
        X["brand_name"] = X.brand_name.str.replace(r"bath & body works", "bath body")
        X["brand_name"] = X.brand_name.str.replace(r"a\|x armani exchange", "axarmaniexchange")

        brand_count_name = X.brand_name.value_counts()[X.brand_name.value_counts() >= 1].loc[
            lambda x: x.index != "missbrand"]
        brand_count_itemdes = X.brand_name.value_counts()[X.brand_name.value_counts() >= 300].loc[
            lambda x: x.index != "missbrand"]
        print("combine brand [name] length: ", len(brand_count_name))
        print("combine brand [item_description] length: ", len(brand_count_itemdes))
        paty_name = r"\b|\b".join(brand_count_name.index.values)
        paty_name = "\\b" + paty_name + "\\b"
        paty_name = re.sub(r"\'", "\\'", paty_name)
        paty_name = re.sub(r'\?', '', paty_name)
        paty_name = re.sub(r'\*', '', paty_name)
        paty_name = re.sub(r'\-', '\-', paty_name)
        paty_name = re.sub(r'\.', '\.', paty_name)
        paty_name = re.sub(r'\’', '\’', paty_name)
        paty_name = re.sub(r'\/', '\/', paty_name)
        paty_name = re.sub(r'\,', '\,', paty_name)
        paty_name = re.sub(r'\!', '\!', paty_name)
        paty_name = re.sub(r'\+', '\+', paty_name)
        self.paty_name = paty_name + '|\\bvs\\b|\\bforever21\\b|\\bmichaelkors\\b|\\bh and m\\b|\\bl\'?oréal\\b|' \
                                     '\\bharley\-davidson\\b|\\btiffany co\.?\\b|\\btiffany\\b|\\blevis?\\b|\\belf\\b|' \
                                     '\\blei\\b|\\bsilver jeans\\b|\\bhollister\\b|\\bdiamond supply\\b|\\bmossimo supply\\b|' \
                                     '\\bchicos?\\b|\\bllbean\\b|\\bana\\b|\\bgilligan o?malley\\b|\\bdelias?\\b|\\bnick jr\\b'

        paty_itemdes = r"\b|\b".join(brand_count_itemdes.index.values)
        paty_itemdes = "\\b" + paty_itemdes + "\\b"
        paty_itemdes = re.sub(r"\'", "\\'", paty_itemdes)
        paty_itemdes = re.sub(r'\?', '\?', paty_itemdes)
        paty_itemdes = re.sub(r'\*', '\*', paty_itemdes)
        paty_itemdes = re.sub(r'\-', '\-', paty_itemdes)
        paty_itemdes = re.sub(r'\.', '\.', paty_itemdes)
        paty_itemdes = re.sub(r'\’', '\’', paty_itemdes)
        paty_itemdes = re.sub(r'\/', '\/', paty_itemdes)
        paty_itemdes = re.sub(r'\,', '\,', paty_itemdes)
        paty_itemdes = re.sub(r'\!', '\!', paty_itemdes)
        paty_itemdes = re.sub(r'\+', '\+', paty_itemdes)
        self.paty_itemdes = paty_itemdes + '|\\bvs\\b|\\bforever21\\b|\\bmichaelkors\\b|\\bh and m\\b|\\bl\'?oréal\\b|' \
                                           '\\bharley\-davidson\\b|\\btiffany co\.?\\b|\\btiffany\\b|\\blevis?\\b|' \
                                           '\\belf\\b|\\blei\\b|\\bsilver jeans\\b|\\bhollister\\b|\\bdiamond supply\\b|' \
                                           '\\bmossimo supply\\b|\\bchicos?\\b|\\bllbean\\b|\\bana\\b|\\bgilligan o?malley\\b|' \
                                           '\\bdelias?\\b|\\bnick jr\\b'
        return self

    def transform(self, X):
        X["item_description"].fillna("missdes", inplace=True)
        X["brand_name"].fillna("missbrand", inplace=True)
        X["name"].fillna("missname", inplace=True)
        X["item_condition_id"].fillna(1, inplace=True)
        X["shipping"].fillna(0, inplace=True)
        X.loc[~X.item_condition_id.isin([1, 2, 3, 4, 5]), "item_condition_id"] = 1
        X.loc[~X.shipping.isin([0, 1]), "shipping"] = 0

        X["item_description"] = X.item_description.str.lower()
        X["name"] = X.name.str.lower()
        X["category_name"] = X.category_name.str.lower()
        X["brand_name"] = X.brand_name.str.lower()

        X["item_description"] = X.item_description.str.replace(r"no description years", "missdes")
        X["item_description"] = X.item_description.str.replace(r"no description [yj]e?t?", "missdes")
        X["item_description"] = X.item_description.str.replace(r"no description\s?\.", "missdes")
        X.loc[X.item_description == "no description", "item_description"] = X.loc[
            X.item_description == "no description", "item_description"].str.replace(r"no description", "missdes")

        X["item_description"] = X.item_description.str.replace(r"6\s+plus", "6plus")
        X["item_description"] = X.item_description.str.replace(r"6s\s+plus", "6splus")
        X["item_description"] = X.item_description.str.replace(r"7\s+plus", "7plus")
        X["item_description"] = X.item_description.str.replace(r"\+", " ")

        X["item_description"] = X.item_description.str.replace(r"iphone\s+7", "iphone7")
        X["item_description"] = X.item_description.str.replace(r"iphone\s+6", "iphone6")
        X["item_description"] = X.item_description.str.replace(r"iphone\s+5", "iphone5")
        X["item_description"] = X.item_description.str.replace(r"iphone\s+4", "iphone4")
        X["item_description"] = X.item_description.str.replace(r"iphone\s+3", "iphone3")
        X["item_description"] = X.item_description.str.replace(r"iphone\s+se", "iphonese")

        X["item_description"] = X.item_description.str.replace(r"ipad\s+pro", "ipadpro")
        X["item_description"] = X.item_description.str.replace(r"ipad\s+air", "ipadair")
        X["item_description"] = X.item_description.str.replace(r"ipad\s+mini", "ipadmini")
        X["item_description"] = X.item_description.str.replace(r"ipod\s+touch", "ipodtouch")
        X["item_description"] = X.item_description.str.replace(r"ipod\s+nano", "ipodnano")

        X["item_description"] = X.item_description.str.replace(r"gbiphone", "gb iphone")
        X["item_description"] = X.item_description.str.replace(r"\s+?gb\b", "gb")

        table1 = str.maketrans("", "", "0.")
        remove1 = lambda m: m.group(0).translate(table1)
        X["item_description"] = X.item_description.str.replace(r"\.0+[a-z]+", remove1)

        def remove2(data):
            al1 = re.findall(r"\d+\.\d*[1-9]+0+", data.group(0))
            al2 = re.findall(r"[a-z]+", data.group(0))
            return al1[0].rstrip("0") + al2[0]

        X["item_description"] = X.item_description.str.replace(r"\d+\.\d*[1-9]+0+[a-z]+", remove2)

        def remove4(data):
            al1 = re.findall(r"\d+(?=\.)", data.group(0))
            al2 = re.findall(r"100%", data.group(0))
            return al1[0] + " " + al2[0]

        X["item_description"] = X.item_description.str.replace(r"\d+\.100%", remove4)

        X["item_description"] = X.item_description.str.replace(r"fl\s?\.?\s?oz", "floz")
        X["item_description"] = X.item_description.str.replace(r"\bw\/", "with ")
        X["item_description"] = X.item_description.str.replace(r"advances? night", "advanced night")

        table5 = str.maketrans("", "", " ")
        remove5 = lambda m: m.group(0).translate(table5)
        X["item_description"] = X.item_description.str.replace(r"\d*?\.?\d+\s\"", remove5)
        X["item_description"] = X.item_description.str.replace(r"\d*?\.?\d+\"?\s?x\s?\d*?\.?\d+\"?\s?x\s?\d*?\.?\d+\"?",
                                                               remove5).str.replace(r"\d*?\.?\d+\"?\s?x\s?\d*?\.?\d+\"?", remove5)
        X["item_description"] = X.item_description.str.replace('\d\s\$', remove5)  #### \$
        X["item_description"] = X.item_description.str.replace(r"\d\sct[ws]?\b", remove5)

        X["item_description"] = X.item_description.str.replace(r"brand[\-\.\,\:\/]?\.?new", "brand new")
        X["item_description"] = X.item_description.str.replace(r"like[•\-\.\,\:\/]\.{0,2}new", "like new")
        X["item_description"] = X.item_description.str.replace(r"newwithtags?", "new with tags").str.replace(
            r"newwithouttags?", "new without tags")

        def findcolon(data):
            al1 = re.findall(r'\d{1,2}\.\d{1,3}|\d{1,2}|1\d{2}', data.group(0))
            return al1[0] + " cooon "

        X["item_description"] = X.item_description.str.replace(
            r'(?<![\d\.])(?:\d{1,2}\.\d{1,3}|\d{1,2}|1\d{2})(?:\s?\")', findcolon)

        X["item_description"] = X.item_description.str.replace(r"play station|playstation", "ps")

        def ps(data):
            al1 = re.findall(r"ps", data.group(0))
            al2 = re.findall(r"\d", data.group(0))
            return al1[0] + al2[0]

        X["item_description"] = X.item_description.str.replace(r"\bps\s\d", ps)

        #### name ####
        X["name"] = X.name.str.replace(r"6\s+plus", "6plus")
        X["name"] = X.name.str.replace(r"6s\s+plus", "6splus")
        X["name"] = X.name.str.replace(r"7\s+plus", "7plus")
        X["name"] = X.name.str.replace(r"\+", " ")

        X["name"] = X.name.str.replace(r"iphone\s+7", "iphone7")
        X["name"] = X.name.str.replace(r"iphone\s+6", "iphone6")
        X["name"] = X.name.str.replace(r"iphone\s+5", "iphone5")
        X["name"] = X.name.str.replace(r"iphone\s+4", "iphone4")
        X["name"] = X.name.str.replace(r"iphone\s+3", "iphone3")
        X["name"] = X.name.str.replace(r"iphone\s+se", "iphonese")

        X["name"] = X.name.str.replace(r"ipad\s+pro", "ipadpro")
        X["name"] = X.name.str.replace(r"ipad\s+air", "ipadair")
        X["name"] = X.name.str.replace(r"ipad\s+mini", "ipadmini")
        X["name"] = X.name.str.replace(r"ipod\s+touch", "ipodtouch")
        X["name"] = X.name.str.replace(r"ipod\s+nano", "ipodnano")

        table3 = str.maketrans("", "", " ")
        remove3 = lambda m: m.group(0).translate(table3)
        X["name"] = X.name.str.replace(r"\d+\s+?gb?\b", remove3)
        X["name"] = X.name.str.replace('\d\s\$', remove3)  #### \$
        X["name"] = X.name.str.replace(r"\d\sct[ws]?\b", remove3)

        X["name"] = X.name.str.replace(r"blu[er\,\- ]{1,2}rays?", "bluray")

        X["name"] = X.name.str.replace(r"play station|playstation", "ps")

        def ps(data):
            al1 = re.findall(r"ps", data.group(0))
            al2 = re.findall(r"\d", data.group(0))
            return al1[0] + al2[0]

        X["name"] = X.name.str.replace(r"\bps\s\d", ps)

        #### brand ####
        X["brand_name"] = X.brand_name.str.replace(r"®|‘|’|™", "")

        X["item_description"] = X.item_description.str.replace(r"\bvictoria\'?s?\s?secret\b", "victoria secret")
        X["name"] = X.name.str.replace(r"\bvictoria\'?s?\s?secret\b", "victoria secret")
        X["brand_name"] = X.brand_name.str.replace(r"victoria\'s secret", "victoria secret")

        X["item_description"] = X.item_description.str.replace(r"\bforever21\b|\bforever twenty one\b", "forever 21")
        X["name"] = X.name.str.replace(r"\bforever21\b|\bforever twenty one\b", "forever 21")

        X["item_description"] = X.item_description.str.replace(r"\bmichaelkors\b", "michael kors")
        X["name"] = X.name.str.replace(r"\bmichaelkors\b", "michael kors")

        X["item_description"] = X.item_description.str.replace(
            r"\bbath\s?&?\s?body\s?works?\b|\bbath\s?&?\s?body\b|\bbath\s?and\s?body\s?works\b|\bbath\s?and\s?body\b",
            "bath body")
        X["name"] = X.name.str.replace(
            r"\bbath\s?&?\s?body\s?works?\b|\bbath\s?&?\s?body\b|\bbath\s?and\s?body\s?works\b|\bbath\s?and\s?body\b",
            "bath body")
        X["brand_name"] = X.brand_name.str.replace(r"bath & body works", "bath body")

        X["item_description"] = X.item_description.str.replace(r"\ba[\/\|]?x armani exchange", "axarmaniexchange")
        X["name"] = X.name.str.replace(r"\ba[\/\|]?x armani exchange", "axarmaniexchange")
        X["brand_name"] = X.brand_name.str.replace(r"a\|x armani exchange", "axarmaniexchange")

        def fillin_name(data):
            bb = re.findall(self.paty_name, data)
            if bb != []:
                return bb[0]
            else:
                return "missbrand"

        def fillin_itemdes(data):
            bb = re.findall(self.paty_itemdes, data)
            if bb != []:
                return bb[0]
            else:
                return "missbrand"

        print("missing brand: ", len(X[X.brand_name == "missbrand"]))
        X.loc[X.brand_name == "missbrand", "brand_name"] = X.loc[X.brand_name == "missbrand", "name"].apply(fillin_name)
        print("missing brand after name: ", len(X[X.brand_name == "missbrand"]))
        X.loc[X.brand_name == "missbrand", "brand_name"] = X.loc[X.brand_name == "missbrand", "item_description"].apply(
            fillin_itemdes)
        print("missing brand after item_description: ", len(X[X.brand_name == "missbrand"]))
        return X


class select_column(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


class split_category(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def split_cat(self, text):
        try:
            return text.split("/")
        except:
            return ("nolabel", "nolabel", "nolabel")

    def transform(self, X):
        X["general_cat"], X["subcat_1"], X["subcat_2"] = zip(*X["category_name"].apply(self.split_cat))
        # print(X.shape)
        return X


class label_bin(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.lb = LabelBinarizer(sparse_output=True)
        self.lb.fit(X)
        self.labels = self.lb.classes_
        return self

    def transform(self, X):
        X[~X.isin(self.labels)] = "missbrand"
        label_bin = self.lb.transform(X).astype(float)
        print('brand shape: ', label_bin.shape)
        print('brand size: {:.2f} mb'.format(sparse_mb(label_bin)))
        return label_bin


class sparse_dummies(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].astype("category")

        shipid = csr_matrix(pd.get_dummies(X, sparse=True).values)
        print('shipid shape: ', shipid.shape)
        return shipid


class add_feature_itemdes(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X.loc[X.item_description.str.contains(r"\bcooon\b"), "cooon"] = 1
        X.loc[X.item_description.str.contains(r"\b10\s?k\b"), "10k"] = 1
        X.loc[X.item_description.str.contains(r"\b12\s?k\b"), "12k"] = 1
        X.loc[X.item_description.str.contains(r"\b14\s?k\b"), "14k"] = 1
        X.loc[X.item_description.str.contains(r"\b18\s?k\b"), "18k"] = 1
        X.loc[X.item_description.str.contains(r"\b22\s?k\b"), "22k"] = 1
        X.loc[X.item_description.str.contains(r"\b24\s?k\b"), "24k"] = 1
        X.loc[X.item_description.str.contains(r"(?<=\d)ct\b"), "ct"] = 1
        X.loc[X.item_description.str.contains(r"(?<=\d)cts\b"), "cts"] = 1
        X.loc[X.item_description.str.contains(r"(?<=\d)ctw\b"), "ctw"] = 1

        X.loc[X.item_description.str.contains(r"\b8\s?gb?\b"), "8gb"] = 1
        X.loc[X.item_description.str.contains(r"\b16\s?gb?\b"), "16gb"] = 1
        X.loc[X.item_description.str.contains(r"\b32\s?gb?\b"), "32gb"] = 1
        X.loc[X.item_description.str.contains(r"\b64\s?gb?\b"), "64gb"] = 1
        X.loc[X.item_description.str.contains(r"\b128\s?gb?\b"), "128gb"] = 1
        X.loc[X.item_description.str.contains(r"\b500\s?gb?\b"), "500gb"] = 1

        X.loc[X.item_description.str.contains(r"brand\s?\-?new"), "brand_new"] = 1
        X.loc[X.item_description.str.contains(r"like\s?\-?new"), "like_new"] = 1
        X.loc[X.item_description.str.contains(r"new\swith\stags?"), "new_with_tags"] = 1
        X.loc[X.item_description.str.contains(r"new\swithout\stags?"), "new_without_tags"] = 1
        X.loc[X.item_description.str.contains(r"\bnwt\b"), "nwt"] = 1
        X.loc[X.item_description.str.contains(r"\bnwot\b"), "nwot"] = 1
        X.loc[X.item_description.str.contains(r"\bbnwt\b"), "bnwt"] = 1
        X.loc[X.item_description.str.contains(r"\bno\b\s?\bflaws?\b"), "no_flaws"] = 1
        X.loc[X.item_description.str.contains(r"\bno\b\s\brips?\b"), "no_rips"] = 1
        X.loc[X.item_description.str.contains(r"\bno\b\s\bscratche?s?\b"), "no_scrath"] = 1
        X.loc[X.item_description.str.contains(r"\bnwb\b"), "nwb"] = 1
        X.loc[X.item_description.str.contains(r"\bnwob\b"), "nwob"] = 1
        X.loc[X.item_description.str.contains(r"\beuc\b"), "euc"] = 1
        X.loc[X.item_description.str.contains(r"\bnever\b\s\bused\b"), "never_used"] = 1
        X.loc[X.item_description.str.contains(r"\bmissing\b"), "missing"] = 1
        X.loc[X.item_description.str.contains(
            r"\bminor\b\s\bimperfections?\b|\bsmall\b\s\bimperfections?\b|\bslight\b\s\bimperfections?\b"), "minor_imperfection"] = 1
        X.loc[X.item_description.str.contains(r"\bno\b\s\bwarranty\b"), "no_warranty"] = 1
        X.loc[X.item_description.str.contains(r"\bwarranty\b"), "warranty"] = 1
        X.loc[X.item_description.str.contains(r"\bdiamonds?\b"), "diamond"] = 1
        X.loc[X.item_description.str.contains(r"\bcrystals?\b"), "crystal"] = 1
        X.loc[X.item_description.str.contains(r"\bgold\b"), "gold"] = 1
        X.loc[X.item_description.str.contains(r"\bsterling\b\s\bsilver\b"), "sterling_silver"] = 1
        X.loc[X.item_description.str.contains(r"\bgrams?\b"), "gram"] = 1
        X.loc[X.item_description.str.contains(r"\bcarat\b"), "carat"] = 1
        X.loc[X.item_description.str.contains(r"\bleather\b"), "leather"] = 1
        X.loc[X.item_description.str.contains(r"\bfaux\b"), "faux"] = 1
        X.loc[X.item_description.str.contains(r"\bsuede\b"), "suede"] = 1
        X.loc[X.item_description.str.contains(r"\bfur\b"), "fur"] = 1
        X.loc[X.item_description.str.contains(r"\bflannel\b"), "flannel"] = 1
        X.loc[X.item_description.str.contains(r"\bvelvet\b"), "velvet"] = 1
        X.loc[X.item_description.str.contains(r"\bceramic\b"), "ceramic"] = 1
        X.loc[X.item_description.str.contains(r"\bwifi\s?only\b"), "wifi_only"] = 1
        X.loc[X.item_description.str.contains(r"\b3g\b"), "3g"] = 1
        X.loc[X.item_description.str.contains(r"\biphone\s?3g\b"), "iphone_3g"] = 1
        X.loc[X.item_description.str.contains(r"\bnew\b"), "new"] = 1
        X.loc[X.item_description.str.contains(r"\bworn\b"), "worn"] = 1
        X.loc[X.item_description.str.contains(r"\bholes?\b"), "hole"] = 1
        X.loc[X.item_description.str.contains(r"\bspots?\b"), "spot"] = 1

        #    print(X.dtypes)
        X.fillna(0, inplace=True)
        addfeature = csr_matrix(X.iloc[:, 1:])
        print('addfeature_item_description shape: ', addfeature.shape)
        print('addfeature_item_description size: {:.2f} mb'.format(sparse_mb(addfeature)))
        return addfeature


class tfidf_itemdes(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.tfidfv = TfidfVectorizer(token_pattern=pattern, ngram_range=(1, 3), max_df=0.7, min_df=3)
        self.tfidfv.fit(X)
        return self

    def transform(self, X):
        tfidf_itemdescription = self.tfidfv.transform(X)
        print('tfidf_item_description shape: ', tfidf_itemdescription.shape)
        print('tfidf_item_description size: {:.2f} mb'.format(sparse_mb(tfidf_itemdescription)))
        return tfidf_itemdescription


class add_feature_name(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X.loc[X.name.str.contains(r"wedding bands?|wedding rings?"), "wedding_ring"] = 1
        X.loc[X.name.str.contains(r"\brings?\b"), "ring"] = 1
        X.loc[X.name.str.contains(r"\b10\s?k\b"), "10k"] = 1
        X.loc[X.name.str.contains(r"\b12\s?k\b"), "12k"] = 1
        X.loc[X.name.str.contains(r"\b14\s?k\b"), "14k"] = 1
        X.loc[X.name.str.contains(r"\b18\s?k\b"), "18k"] = 1
        X.loc[X.name.str.contains(r"\b22\s?k\b"), "22k"] = 1
        X.loc[X.name.str.contains(r"\b24\s?k\b"), "24k"] = 1
        X.loc[X.name.str.contains(r"(?<=\d)ct\b"), "ct"] = 1
        X.loc[X.name.str.contains(r"(?<=\d)cts\b"), "cts"] = 1
        X.loc[X.name.str.contains(r"(?<=\d)ctw\b"), "ctw"] = 1

        X.loc[X.name.str.contains(r"\b8\s?gb?\b"), "8gb"] = 1
        X.loc[X.name.str.contains(r"\b16\s?gb?\b"), "16gb"] = 1
        X.loc[X.name.str.contains(r"\b32\s?gb?\b"), "32gb"] = 1
        X.loc[X.name.str.contains(r"\b64\s?gb?\b"), "64gb"] = 1
        X.loc[X.name.str.contains(r"\b128\s?gb?\b"), "128gb"] = 1
        X.loc[X.name.str.contains(r"\b500\s?gb?\b"), "500gb"] = 1

        X.loc[X.name.str.contains(r"\bps1\b"), "ps1"] = 1
        X.loc[X.name.str.contains(r"\bps2\b"), "ps2"] = 1
        X.loc[X.name.str.contains(r"\bps3\b"), "ps3"] = 1
        X.loc[X.name.str.contains(r"\bps4\b"), "ps4"] = 1

        X.loc[X.name.str.contains(r"brand\s?\-?new"), "brand_new"] = 1
        X.loc[X.name.str.contains(r"new\swith\stags?"), "new_with_tags"] = 1
        X.loc[X.name.str.contains(r"new\swithout\stags?"), "new_without_tags"] = 1
        X.loc[X.name.str.contains(r"\bnwt\b"), "nwt"] = 1
        X.loc[X.name.str.contains(r"\bnwot\b"), "nwot"] = 1
        X.loc[X.name.str.contains(r"\bbnwt\b"), "bnwt"] = 1
        X.loc[X.name.str.contains(r"\bno\b\s?\bflaws?\b"), "no_flaws"] = 1
        X.loc[X.name.str.contains(r"\bno\b\s\bcracks?\b"), "no_crack"] = 1
        X.loc[X.name.str.contains(r"\bno\b\s\bscratche?s?\b"), "no_scrath"] = 1
        X.loc[X.name.str.contains(r"\bnew\b\s\bwith\b\s\bboxe?s?\b"), "new_with_box"] = 1
        X.loc[X.name.str.contains(r"\bnew\b\s\bwithout\b\s\bboxe?s?\b"), "new_without_box"] = 1
        X.loc[X.name.str.contains(r"\bnwb\b"), "nwb"] = 1
        X.loc[X.name.str.contains(r"\bnwob\b"), "nwob"] = 1
        X.loc[X.name.str.contains(r"\bnever\b\s\bworn\b"), "never_worn"] = 1
        X.loc[X.name.str.contains(r"\bnever\b\s\bused\b"), "never_used"] = 1
        X.loc[X.name.str.contains(r"\bscratched\b"), "scratched"] = 1
        X.loc[X.name.str.contains(r"\bswatched\b"), "swatched"] = 1
        X.loc[X.name.str.contains(r"\bunopened\b"), "unopened"] = 1
        X.loc[X.name.str.contains(r"\bno\b\s\bholes?\b"), "no_hole"] = 1
        X.loc[X.name.str.contains(r"\bseale?d?\b"), "sealed"] = 1
        X.loc[X.name.str.contains(r"\bmissing\b"), "missing"] = 1
        X.loc[X.name.str.contains(r"\bpic\b|\bpictures?d?\b|\bimages?\b"), "picture"] = 1
        X.loc[X.name.str.contains(r"\bobo\b|\bor\b\s\bbest\b\s\boffer\b"), "obo"] = 1
        X.loc[X.name.str.contains(r"\bbundle\b"), "bundle"] = 1
        X.loc[X.name.str.contains(r"\bwarranty\b"), "warranty"] = 1
        X.loc[X.name.str.contains(r"\bsmoke\b\s\bfree\b"), "smoke_free"] = 1
        X.loc[X.name.str.contains(r"\bdiamonds?\b"), "diamond"] = 1
        X.loc[X.name.str.contains(r"\bgold\b"), "gold"] = 1
        X.loc[X.name.str.contains(r"\bsterling\b\s\bsilver\b"), "sterling_silver"] = 1
        X.loc[X.name.str.contains(r"\bauthentic\b"), "authentic"] = 1
        X.loc[X.name.str.contains(r"\bauthenticity\b"), "authenticity"] = 1
        X.loc[X.name.str.contains(r"\brare\b"), "rare"] = 1
        X.loc[X.name.str.contains(r"\bgrams?\b"), "gram"] = 1
        X.loc[X.name.str.contains(r"\bcarat\b"), "carat"] = 1
        X.loc[X.name.str.contains(r"100%"), "100%"] = 1
        X.loc[X.name.str.contains(r"\bmedium\b|\bmed\b|\bm\b"), "medium"] = 1
        X.loc[X.name.str.contains(r"\blarge\b|\bbig\b|\bl\b|\bxl\b|\bxxl\b|\b2xl\b"), "large"] = 1
        X.loc[X.name.str.contains(r"\bleather\b"), "leather"] = 1
        X.loc[X.name.str.contains(r"\bfaux\b"), "faux"] = 1
        X.loc[X.name.str.contains(r"\bsuede\b"), "suede"] = 1
        X.loc[X.name.str.contains(r"\bfur\b"), "fur"] = 1
        X.loc[X.name.str.contains(r"\bpolyester\b"), "polyester"] = 1
        X.loc[X.name.str.contains(r"\bfabric\b"), "fabric"] = 1
        X.loc[X.name.str.contains(r"\bceramic\b"), "ceramic"] = 1
        X.loc[X.name.str.contains(r"\binch\b"), "inch"] = 1
        X.loc[X.name.str.contains(r"\bcm\b"), "cm"] = 1
        X.loc[X.name.str.contains(r"\bnickel\b\s\bfree\b"), "nickle_free"] = 1
        X.loc[X.name.str.contains(r"\bwifi\s?only\b"), "wifi_only"] = 1
        X.loc[X.name.str.contains(r"\biphone\s?3g\b"), "iphone_3g"] = 1
        X.loc[X.name.str.contains(r"\bnew\b"), "new"] = 1
        X.loc[X.name.str.contains(r"\bremoved\b"), "removed"] = 1
        X.loc[X.name.str.contains(r"\bflaws?\b"), "flaw"] = 1
        X.loc[X.name.str.contains(r"\bdefects?\b"), "defect"] = 1
        X.loc[X.name.str.contains(r"\bcracks?\b"), "crack"] = 1
        X.loc[X.name.str.contains(r"\bscratche?s?\b"), "scrath"] = 1
        X.loc[X.name.str.contains(r"\bperfect\b"), "perfect"] = 1
        X.loc[X.name.str.contains(r"\bworn\b"), "worn"] = 1
        X.loc[X.name.str.contains(r"\bnoticeable\b"), "noticeable"] = 1
        X.loc[X.name.str.contains(r"\bused\b"), "used"] = 1
        X.loc[X.name.str.contains(r"\bopened\b"), "opened"] = 1
        X.loc[X.name.str.contains(r"\bholes?\b"), "hole"] = 1
        X.loc[X.name.str.contains(r"\bspots?\b"), "spot"] = 1
        X.loc[X.name.str.contains(r"\bfree\b"), "free"] = 1
        X.loc[X.name.str.contains(r"\bno\b"), "no"] = 1
        X.loc[X.name.str.contains(r"\bstill\b"), "still"] = 1
        X.loc[X.name.str.contains(r"\bnever\b"), "never"] = 1
        X.loc[X.name.str.contains(r"\boriginally\b"), "originally"] = 1
        X.loc[X.name.str.contains(r"\bmsrp\b"), "msrp"] = 1
        #    print(X.dtypes)
        X.fillna(0, inplace=True)
        addfeature = csr_matrix(X.iloc[:, 1:])
        print('addfeature_name shape: ', addfeature.shape)
        print('addfeature_name size: {:.2f} mb'.format(sparse_mb(addfeature)))
        return addfeature


class tfidf_name(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.tfidfv = TfidfVectorizer(token_pattern=pattern, ngram_range=(1, 3), max_df=0.7, min_df=3)
        self.tfidfv.fit(X)
        return self

    def transform(self, X):
        tfidf_name = self.tfidfv.transform(X)
        print('tfidf_name shape: ', tfidf_name.shape)
        print('tfidf_name size: {:.2f} mb'.format(sparse_mb(tfidf_name)))
        return tfidf_name


class tfidf_brand_category(BaseEstimator, TransformerMixin):
    def __init__(self, gram=3, maxdf=1.0, mindf=1):
        self.gram = gram
        self.maxdf = maxdf
        self.mindf = mindf

    def fit(self, X, y=None):
        self.tfidfv = TfidfVectorizer(token_pattern=pattern111, ngram_range=(1, self.gram), max_df=self.maxdf,
                                      min_df=self.mindf)
        self.tfidfv.fit(X)
        return self

    def transform(self, X):
        tfidf_brand_category = self.tfidfv.transform(X)
        print('tfidf_brand_category shape: ', tfidf_brand_category.shape)
        print('tfidf_brand_category size: {:.2f} mb'.format(sparse_mb(tfidf_brand_category)))
        return tfidf_brand_category


full_pipe = Pipeline([
    ('basic_cleaning', cleaning()),
    ('split_category', split_category()),

    ('union', FeatureUnion(transformer_list=[
        ('item_description_addfeature', Pipeline([
            ('selector', select_column(cols="item_description")),
            ('addfeature_itemdes', add_feature_itemdes()),
        ])),
        ('item_description_tfidf', Pipeline([
            ('selector', select_column(cols="item_description")),
            ('tfidf_itemdes', tfidf_itemdes()),
        ])),
        ('name_addfeature', Pipeline([
            ('selector', select_column(cols="name")),
            ('addfeature_name', add_feature_name()),
        ])),
        ('name_tfidf', Pipeline([
            ('selector', select_column(cols="name")),
            ('tfidf_name', tfidf_name()),
        ])),
        ('ship_id', Pipeline([
            ('selector', select_column(cols=['item_condition_id', 'shipping'])),
            ('sparse_dummies', sparse_dummies()),
        ])),
        ('LabelBinarize', Pipeline([
            ('selector', select_column(cols=['brand_name'])),
            ('labelbin', label_bin()),
        ])),
        ('general_cat_tfidf', Pipeline([
            ('selector', select_column(cols='general_cat')),
            ('tfidf_general_cat', tfidf_brand_category()),
        ])),
        ('subcat_1_tfidf', Pipeline([
            ('selector', select_column(cols='subcat_1')),
            ('tfidf_subcat_1', tfidf_brand_category()),
        ])),
        ('subcat_2_tfidf', Pipeline([
            ('selector', select_column(cols='subcat_2')),
            ('tfidf_subcat_2', tfidf_brand_category()),
        ])),
    ])),
])

print('start pipeline')
start_time = time.time()
train = full_pipe.fit_transform(train)
print("full_pipe fit_time: ", time.time() - start_time)

start_time = time.time()
model_1 = AdaGradRegressor(random_state=42, alpha=2e-6, n_iter=30, eta=0.1)
model_1.fit(train, y)
print("model_1 fit_time: ", time.time() - start_time)

start_time = time.time()
model_2 = SDCARegressor(random_state=42, max_iter=30, alpha=2e-6, tol=0.0001)
model_2.fit(train, y)
print("model_2 fit_time: ", time.time() - start_time)

del train
gc.collect()

print('start predicting')


def load_test():
    for df in pd.read_csv('../input/test_stg2.tsv', sep='\t', chunksize=700000):
        yield df


pred = []
for data in load_test():
    data = full_pipe.transform(data)
    pred.append((model_1.predict(data) + model_2.predict(data)) / 2)

pred = np.expm1(np.concatenate(pred))
pred[pred < 3] = 3.0

result = pd.read_csv('../input/test_stg2.tsv', sep='\t', usecols=['test_id'])
result["price"] = pred.reshape(-1, 1)
result.to_csv("submission.csv", index=False)
