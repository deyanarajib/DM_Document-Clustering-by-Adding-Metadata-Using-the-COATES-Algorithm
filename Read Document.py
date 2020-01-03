from nltk.tokenize import word_tokenize as token

with open('tes.txt') as f:
    reader = f.readlines()
reader = [x.strip() for x in reader]

book = ['Book'+' '+str(i)+' :' for i in range(1,100)]

booktitle=[];secttitle=[];hadithnumb=[];hadith=[];indeks=[]
for idx,i in enumerate(reader):
    if ' '.join(token(i)[0:3]) in book:
        booktitle.append(i)
    try:
        if token(i)[0] == 'SECTION':
            secttitle.append(i)
    except:
        continue
    if 'Number' in token(i):
        indeks.append(idx)
        idx = i.index(':')
        hadithnumb.append(i[0:idx+1])

hadith=[]
for i in range(len(indeks)):
    if i == len(indeks)-1:
        hadith.append(' '.join(reader[indeks[i]::]))
    else:
        hadith.append(' '.join(reader[indeks[i]:indeks[i+1]]))

data=[]
for i in hadith:
    idx = i.index(':')
    data.append(i[idx+2::])

sanad=[];matan=[]
for i in data:
    try:
        idx = i.index('that')
        sanad.append(i[0:idx])
        matan.append(i[idx+5::])
    except:
        continue

for idx,i in enumerate(sanad):
    if 'me' in token(i):
        sanad[idx] = sanad[idx].replace('me','MalikBinAnas')


import pandas as pd
x = pd.DataFrame(sanad)
x.to_csv('sanad.csv', index=False, header=False)
x = pd.DataFrame(matan)
x.to_csv('matan.csv', index=False, header=False)
