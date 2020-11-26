from bs4 import BeautifulSoup
import pandas as pd
from hanziconv import HanziConv
import uuid

MAX_LENGTH = 50

def convert():
    # if encounters UnicodeDecode warning, go to that sgml file and type a space somewhere safe lol and you can remove it
    passage_dict = {}
    data_dict = {}
    passages, correcteds = {}, {}

    f = open("data/sighan7csc_release1.0 2/SampleSet/Bakeoff2013_SampleSet_WithError_00001-00350.txt", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for d in soup.find_all('doc'):
        doc_id = d['nid']
        
        wrong = HanziConv.toSimplified(d.find_all('wrong')[0].text)
        correction = HanziConv.toSimplified(d.find_all('correct')[0].text) 
        passage = HanziConv.toSimplified(d.find_all('p')[0].text)
        if len(passage) <= MAX_LENGTH:
            corrected = passage.replace(wrong, correction)
            passages[doc_id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[doc_id] = corrected.replace(" ", "")
    f.close()

    f = open("data/sighan7csc_release1.0 2/SampleSet/Bakeoff2013_SampleSet_WithoutError_10001-10350.txt", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for d in soup.find_all('doc'):
        doc_id = d['nid']
        
        passage = HanziConv.toSimplified(d.find_all('p')[0].text)
        if len(passage) <= MAX_LENGTH:
            passages[doc_id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[doc_id] = passage.replace(" ", "")
    f.close()

    f = open("data/sighan8csc_release1.0/Training/SIGHAN15_CSC_A2_Training.sgml", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for p in soup.find_all('passage'):
        passage_dict[p['id']] = HanziConv.toSimplified(p.text)
    for m in soup.find_all('mistake'):
        id = m['id']

        wrong = HanziConv.toSimplified(m.find_all('wrong')[0].text)
        correction = HanziConv.toSimplified(m.find_all('correction')[0].text) 
        passage = passage_dict[id]
        if len(passage) <= MAX_LENGTH:
            corrected = passage.replace(wrong, correction)
            passage_dict[id] = corrected.replace(" ", "") # there might be more mistakes to this passage so we need to update the passage
            if id not in passages:
                passages[id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[id] = corrected.replace(" ", "")
    f.close()

    f = open("data/sighan8csc_release1.0/Training/SIGHAN15_CSC_B2_Training.sgml", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for p in soup.find_all('passage'):
        passage_dict[p['id']] = HanziConv.toSimplified(p.text)
    for m in soup.find_all('mistake'):
        id = m['id']

        wrong = HanziConv.toSimplified(m.find_all('wrong')[0].text)
        correction = HanziConv.toSimplified(m.find_all('correction')[0].text) 
        passage = passage_dict[id]
        if len(passage) <= MAX_LENGTH:
            corrected = passage.replace(wrong, correction)
            passage_dict[id] = corrected.replace(" ", "") # there might be more mistakes to this passage so we need to update the passage
            if id not in passages:
                passages[id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[id] = corrected.replace(" ", "")
    f.close()

    f = open("data/clp14csc_release1.1/Training/B1_training.sgml", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for p in soup.find_all('passage'):
        passage_dict[p['id']] = HanziConv.toSimplified(p.text)
    for m in soup.find_all('mistake'):
        id = m['id']

        wrong = HanziConv.toSimplified(m.find_all('wrong')[0].text)
        correction = HanziConv.toSimplified(m.find_all('correction')[0].text) 
        passage = passage_dict[id]
        if len(passage) <= MAX_LENGTH:
            corrected = passage.replace(wrong, correction)
            passage_dict[id] = corrected.replace(" ", "") # there might be more mistakes to this passage so we need to update the passage
            if id not in passages:
                passages[id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[id] = corrected.replace(" ", "")
    f.close()
    
    f = open("data/clp14csc_release1.1/Training/C1_training.sgml", "r")
    soup = BeautifulSoup(f, 'html.parser')
    
    for p in soup.find_all('passage'):
        passage_dict[p['id']] = HanziConv.toSimplified(p.text)
    for m in soup.find_all('mistake'):
        id = m['id']

        wrong = HanziConv.toSimplified(m.find_all('wrong')[0].text)
        correction = HanziConv.toSimplified(m.find_all('correction')[0].text) 
        passage = passage_dict[id]
        if len(passage) <= MAX_LENGTH:
            corrected = passage.replace(wrong, correction)
            passage_dict[id] = corrected.replace(" ", "") # there might be more mistakes to this passage so we need to update the passage
            if id not in passages:
                passages[id] = passage.replace(" ", "") # the very first untampered source sentence
            correcteds[id] = corrected.replace(" ", "")
    f.close()

    """f = open("data/pycorrector_data.txt", "r")
    for x in f:
        [src, target] = x.split('\t')
        pid = uuid.uuid4()
        src, target = src.replace(" ", "").strip(), target.replace(" ", "").strip()
        if (len(src) <= MAX_LENGTH and len(target) <= MAX_LENGTH):
            passages[pid] = src
            correcteds[pid] = target"""

    p, c = [], []
    for id in passages.keys():
        if len(passages[id]) == len(correcteds[id]):
            p.append(passages[id])
            c.append(correcteds[id])


    dataset = pd.DataFrame(columns=['source','reference'])
    dataset['source'] = p
    dataset['reference'] = c
    dataset.to_csv("data/sigha50.csv", index=False)

if __name__ == "__main__":
    convert()