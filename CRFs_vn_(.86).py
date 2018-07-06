import pandas as pd
import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
import re
# Uses pandas
# data = pd.read_csv('D:\\Machine_learning\\NLP\\ner_data.txt', sep='\t', header=None, names=['word', 'NE'])
# word = data['word'].values.tolist()
# # print('check: ',word[138510])
# NE = data['NE'].values.tolist()
# replace '_' in words, separate word


def separate_word(word, NE):
    i = 0
    while not (i == (len(word)-1)):
        if '_' in word[i]:
            listword = word[i].split('_')
            # print('listword: ',listword)
            listTag = []
            choseTag = {'B-PER':'I-PER','B-LOC':'I-LOC','B-ORG':'I-ORG','B-PRO':'I-PRO'}
            Tag = choseTag.get(NE[i],'O')
            for idx in range(0,len(listword)):
                if idx ==0:
                    listTag.append(NE[i])
                else:
                    listTag.append(Tag)
            # print(listTag)
            NE.pop(i)
            word.pop(i)
            word[i:i] =  listword
            NE[i:i] = listTag
            i += len(listTag)
            continue
        i +=1
    return (word,NE)


# (word,NE) = separateword(word,NE)
# print(word[:100])
# print(NE[:50])
# train_data = [(Word,Ne) for Word,Ne in zip(word, NE)]
data2 = []
with open('D:\\Machine_learning\\NLP\\ner_data.txt',encoding='utf-8') as file:
    line = file.read().split('\n')
    # temp = [l.split('\t') for l in line]
    # for i in range(0,len(temp)):
    #     data2.append(tuple(temp[i]))
    data2 = [tuple(l.split('\t'))for l in line]
train_data = data2
print(type(data2))
# print(data2[:60])
sent = []
train_sents = []
# separate sentence
for word_tag in train_data:
    if(word_tag[0] ==''):
        continue
    if word_tag[0] == '.':
        train_sents.append(sent)
        sent = []
        continue
    sent.append(word_tag)
# print(train_sents[:2])
print('total: ',len(train_sents))


print('--Extract feature \n')


def word2feature(sent, i):
    word = sent[i][0]
    #print(sent[i],'\n')
    Tag = sent[i][1]
    features = [
        'bias',
        'word.lower='+word.lower(),
        'word[-3:]='+word[-3:],
        'word[-2:]='+word[-2:],
        'word.isupper=%s'%word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.hashyphen=%s' % word.find('-'),
        # 'tag=' + Tag,
        # 'tag[:2]=' + Tag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        Tag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            # '-1:tag=' + Tag1,
            # '-1:tag[:2]=' + Tag1[:2],
        ])
    else:
        features.append('BOS')
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        Tag1 = sent[i+1][1]
        # print(word1,'_',Tag1)

        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            # '+1:tag=' + Tag1,
            # '+1:tag[:2]=' + Tag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2feature(sent):
    return [word2feature(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def getdataset(data, num, random_list):
    data_set = []
    if num > 0:
        data_set = [data[i] for i in random_list[:num]]
    else:
        if num < 0:
            data_set = [data[i] for i in random_list[num:]]
        else:
            data_set = [data[i] for i in random_list[0]]
    return data_set


randomList = np.random.permutation(len(train_sents))
x_set1 = [sent2feature(s) for s in train_sents[:2000]]
y_set1 = [sent2labels(s) for s in train_sents[:2000]]

x_set2 = [sent2feature(s) for s in train_sents[2000:4000]]
y_set2 = [sent2labels(s) for s in train_sents[2000:4000]]

x_set3 = [sent2feature(s) for s in train_sents[4000:7000]]
y_set3 = [sent2labels(s) for s in train_sents[4000:7000]]

x_set4 = [sent2feature(s) for s in train_sents[7000:9000]]
y_set4 = [sent2labels(s) for s in train_sents[7000:9000]]

x_set5 = [sent2feature(s) for s in train_sents[9000:]]
y_set5 = [sent2labels(s) for s in train_sents[9000:]]

x_append = []
x_append.append(x_set1)
x_append.append(x_set2)
x_append.append(x_set3)
x_append.append(x_set4)
x_append.append(x_set5)

y_append = []
y_append.append(y_set1)
y_append.append(y_set2)
y_append.append(y_set3)
y_append.append(y_set4)
y_append.append(y_set5)

# print('xtrain: ',X_train[0])
# print('xtest:  ',X_test[0])


'''---Evaluate the model---'''
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()

    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    # print('u_true_combined; \n',y_true)
    # print(len(y_true))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    # print('y_pre',y_pred)
    tagset = set(lb.classes_)-{'O'}-{'B-PRO'}-{'I-PRO'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    # print(type(lb))
    # print('label: \n',lb)
    # print('tagset: \n',tagset)
    # print('lb.class: \n',lb.classes_)
    # print('class_indice: \n',class_indices)
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(5):
    for k in range(5):
        if(k!=i):
            X_train.extend(x_append[k])
            y_train.extend(y_append[k])
        else:
            X_test.extend(x_append[i])
            y_test.extend(y_append[i])
    # print(X_train)
    print('number of train data %d: '%i,len(X_train))
    print('number of test data %d: '%i,len(X_test))

    '''---Train the model---'''

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    # use L-BFGS traing Algorithm(default) with Elastic Net regularization
    # set parameter
    trainer.set_params({
        'c1': 1., #coefficient for L1 penalty
        'c2': 1e-3,# for L2
        'max_iterations': 50, # stop earlier
        'feature.possible_transitions': True
    })
    # print(trainer.params())
    trainer.train('vn_test.crfsuite')
    # print(trainer.logparser.last_iteration)

    '''---Prediction---'''
    tagger = pycrfsuite.Tagger()
    tagger.open('vn_test.crfsuite')
    example_ = train_sents[-3:]
    # print('1', example_[0])
    # example_ = [('xin', 'O'), ('cấp', 'O'), ('Giấy', 'O'), ('Chứng_Nhận', 'O'), ('tại', 'O'), ('UBND', 'B-ORG'),
    #             ('thành_phố', 'I-ORG')]
    # print(' '.join(sent2tokens(example_)), '\n')
    # print('Predict: \n', tagger.tag(sent2feature(example_)))
    # print('Correct:   \n', sent2labels(example_))

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test,y_pred))
    X_train.clear()
    X_test.clear()
    y_train.clear()
    y_test.clear()
