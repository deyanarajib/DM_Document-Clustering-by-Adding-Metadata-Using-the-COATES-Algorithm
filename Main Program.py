import pandas as pd

with open('sanad.csv') as f:
    reader = f.readlines()
reader = [x.strip() for x in reader]
sanad=[]
for row in reader:
    sanad.append(row)
with open('matan.csv') as f:
    reader = f.readlines()
reader = [x.strip() for x in reader]
matan=[]
for row in reader:
    matan.append(row)

import random
import string
import itertools
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize as token
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import rankdata
from scipy.spatial.distance import squareform,pdist
import warnings
warnings.simplefilter('ignore')

meaningless = stopwords.words('english')+list(string.punctuation)+['``',"''"]+[str(i) for i in range(100)]
stemmer = SnowballStemmer('english')

dmatan=[]
for i in matan:
    temp=[]
    for j in token(i):
        word = stemmer.stem(str.lower(j))
        if word not in meaningless and "'" not in j and '.' not in j:
            temp.append(word)
    dmatan.append(temp)

dsanad=[]
for i in sanad:
    temp=[]
    for j in token(i):
        word = stemmer.stem(str.lower(j))
        if word not in meaningless and "'" not in j and '.' not in j:
            temp.append(word)
    dsanad.append(temp)

vectorizer = TfidfVectorizer()
alldata = vectorizer.fit_transform([' '.join(i) for i in dmatan]).todense()
alldata = np.squeeze(np.asarray(alldata))

#x = pd.DataFrame(alldata)
#x.to_csv('tfidf.csv', index=False, header=False)

def Euclid(A,B):
    return np.linalg.norm(A-B)
def Sumvec(A,B):
    return A+B
def CountLink(A,B):
    return np.dot(A,B)
def FindSimi(A,B):
    return simbank[A[0]][B[0]]
def FindInitCent(simbank,neghbrs,teta,K,Nplus):
    idxcand = np.argsort(neghbrs.sum(0))[-(K+Nplus):][::-1]
    cndidte = neghbrs[idxcand]
    linkcnd = pdist(cndidte,metric=CountLink)
    simicnd = pdist(idxcand.reshape(K+Nplus,1),metric=FindSimi)
    ranklnk = rankdata(linkcnd,method='dense')
    ranksim = rankdata(simicnd,method='dense')
    ranksum = ranklnk+ranksim
    rankcom =[]; idxcomb=[]
    for i in itertools.combinations(range(len(idxcand)),K):
        idxcomb.append(i); temp = 0
        for j in itertools.combinations(i,2):
            temp = temp + squareform(ranksum)[j[0]][j[1]]
        rankcom = np.append(rankcom,temp)
    return [idxcand[i] for i in idxcomb[np.argmin(rankcom)]]

def FindCluster(data,cent):
    x=[]
    for i in data:
        x.append(np.argmin([Euclid(i,j) for j in cent]))
    return x

K = 40; teta = 0.9; Nplus=2
simbank = 1-squareform(1-pdist(alldata,metric='cosine'))
neghbrs = np.where(simbank >= teta,1,0)
initc = FindInitCent(simbank,neghbrs,teta,K,Nplus)

cent1 = alldata[initc]
clus1 = FindCluster(alldata,cent1)

convergent = False; maxiter=1
while not convergent:
    cent2=[]
    for i in range(K):
        result = alldata[np.isin(clus1,i)].sum(axis=0)/clus1.count(i)
        cent2.append(result)

    clus2 = FindCluster(alldata,cent2)

    if clus1 == clus2:
        convergent = True
    else:
        clus1 = clus2
        maxiter += 1

clus_first_iter = clus2

print('\nCLUSTER FIRST ITERATION')
for i in range(K):
    print('CLUSTER',i+1,':',np.arange(len(alldata))[np.isin(clus_first_iter,i)])

x = pd.DataFrame(clus_first_iter)
x.to_csv('ClusterFirstIteration.csv', index=False, header=False)

print('JUMLAH ELEMEN CLUSTER 1 s/d 40 FIRST ITERATION:')
for i in range(K):
    print(clus_first_iter.count(i),' ',end='')
print('\n')

dictclus=[]
for i in range(K):
    dictclus.append(list(set(np.hstack(np.array(dmatan)[np.isin(clus_first_iter,i)]))))

frm=[]; frj=[]
for idx1,i in enumerate(dsanad):
    temp=[]
    for idx2,j in enumerate(dictclus):
        temp.append(len(set(i)-set(j)))
    frj.append(temp)
    frm.append(sum(temp))

prj=[]; giny=[]
for i in range(len(frj)):
    if frm[i] != 0:
        prj.append(np.array(frj[i])/frm[i])
    else:
        prj.append([0]*len(frj[i]))
    giny.append(sum(np.array(prj[i])**2))

ginyavg = np.average(giny)

ri=[]
for idx,i in enumerate(giny):
    if i < ginyavg:
        ri.append(idx)

prob=[]; rifix=[]
for idx,i in enumerate (dsanad):
    if idx not in ri:
        continue
    temp=[]
    for j in dictclus:
        temp.append(len(set(i)&set(j)))
    if max(temp) != 0:
        jmlmax = temp.count(max(temp))
        if jmlmax == 1:
            prob.append(temp.index(max(temp)))
            rifix.append(idx)
        else:
            if max(temp) != temp[clus_first_iter[idx]]:
                populate = np.arange(K)[np.isin(temp,max(temp))].tolist()
                prob.append(random.sample(populate,1)[0])
                rifix.append(idx)

clus_second_iter = [i for i in clus_first_iter]
count = 0
for i in range(len(clus_second_iter)):
    if i in rifix:
       clus_second_iter[i] = prob[count]
       count += 1

print('\nCLUSTER FIRST ITERATION')
for i in range(K):
    print('CLUSTER',i+1,':',np.arange(len(alldata))[np.isin(clus_second_iter,i)])

x = pd.DataFrame(clus_second_iter)
x.to_csv('ClusterSecondIteration.csv', index=False, header=False)
print('JUMLAH ELEMEN CLUSTER 1 s/d 40 SECOND ITERATION:')
for i in range(K):
    print(clus_second_iter.count(i),' ',end='')
print('\n')
       
for i in range(len(clus_first_iter)):
    if clus_first_iter[i] != clus_second_iter[i]:
        print('Doc',i+1,'Move from Cluster',clus_first_iter[i]+1,'to Cluster',clus_second_iter[i]+1)
