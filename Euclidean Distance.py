from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import math

def EuclideanDistance (centroid,TFIDF):
    temp=0;temp2=[];jarak=[]
    for i in range(0,len(centroid)):
        for j in range(0,len(TFIDF)):
            for k in range(0,len(TFIDF[0])):
                temp = temp + math.pow(centroid[i][k]-TFIDF[j][k],2)
            temp2.append(math.sqrt(temp))
            temp=0
        jarak.append(temp2)
        temp2=[]
    euclid=[]
    temp = []
    for i in range(0,len(jarak[0])):
        for j in range(0,len(jarak)):
            temp.append(jarak[j][i])
        euclid.append(temp)
        temp=[]
    return euclid

def CosineSimilarity (centroid,TFIDF):
    temp4=[];Cosine=[]
    for i in range(0,len(TFIDF)):
        for j in range (0,len(centroid)):
            temp1=sum([TFIDF[i][k]*centroid[j][k] for k in range(0,len(TFIDF[0]))])
            temp2=math.sqrt(sum([math.pow(TFIDF[i][k],2) for k in range(0,len(TFIDF[0]))]))
            temp3=math.sqrt(sum([math.pow(centroid[j][k],2) for k in range(0,len(TFIDF[0]))]))
            temp4.append(temp1/(temp2*temp3))
        Cosine.append(temp4)
        temp4=[]
    return Cosine

def getIndeks(X,Similar,TFIDF):
    indeks=[]
    for i in range(0,len(TFIDF)):
        if X == 0:
            indeks.append(Similar[i].index(min(Similar[i])))
        else:
            indeks.append(Similar[i].index(max(Similar[i])))
    return indeks

def readDocument(a,b):
    doc=[]
    for i in range(a,b):
        wordList = [str(i+1),".txt"]
        sentence = ""
        for i in wordList:
            sentence += i
        temp2=open(sentence, "r").read()
        doc.append(temp2)
    return doc

def readDocumentm(a,b):
    doc=[]
    for i in range(a,b):
        wordList = [str(i+1),".txt"]
        sentence = "m"
        for i in wordList:
            sentence += i
        temp2=open(sentence, "r").read()
        doc.append(temp2)
    return doc

def PreProcessing(corpus):
    doc=[];token=[];
    stemmer=PorterStemmer()
    stop_words=stopwords.words('english')+list(string.punctuation)
    count=0
    for i in corpus:
        count = count+1
        temp=word_tokenize(i)#Tokenizing
        for j in temp:
            if j not in stop_words: #Hilangkan Stop Words
                token.append(stemmer.stem(j)) #Stemming&CaseFolding
        doc.append(token)
        token=[]
    return doc

def getDictionary(doc):
    dictionary=[]
    for i in doc:
        for j in i:
            if not j in dictionary: #Hilangkan Duplikasi
                dictionary.append(j)
    return dictionary

def getNewCentroid(indeks,TFIDF,K):
    jmlclus=[]; temp2=[]
    for i in range(0,K):
        jmlclus.append(indeks.count(i))

    idx=[]
    for i in range(0,len(indeks)):
        idx.append(indeks[i])

    temp=[0]*len(TFIDF[0])
    centroid=[]
    for i in range(0,K):
        for j in range(0,jmlclus[i]):
            temp=[x + y for x, y in zip(temp,TFIDF[idx.index(i)][:])]
            idx[idx.index(i)]=-1
        print('  Finding Centroid',i+1,end='')
        if jmlclus[i]!=0:
            temp2=[z/jmlclus[i] for z in temp]
        centroid.append(temp2)
        temp=[0]*len(TFIDF[0])
        print('...Done!')
    return centroid


def printCluster(indeks,K):
    print('')
    for i in range(0,K):
        print('CLUSTER',i+1,':',end='')
        for j in range(0,len(indeks)):
            if indeks[j] == i:
                print(' Doc',j+1,end='')
        print('')
    print('')

def printme(A,B):
    if B == 1:
        print('  [',A[0],A[1],A[2],'...',A[len(A)-1],']')
        print('')
    else:
        print('  [',A[0][0],A[0][1],A[0][2],'...',A[0][len(A[0])-1],']')
        for i in range(0,3): print('     .')
        print('  [',A[0][B-1],A[1][B-1],A[2][B-1],'...',A[B-1][len(A[0])-1],']\n')
        
rawcont = readDocument(0,500)
rawmeta = readDocumentm(0,500)

print('==============')
print('Term Weightned')
print('==============')

print('1.Pre-Processing',end='')
content = PreProcessing(rawcont)
auxiliary = PreProcessing(rawmeta)
print('...Done')

print('2.Finding Dictionary')
dictcont = getDictionary(content)
print('  Content Dictionary = Vector 1 x',len(dictcont))
printme(A=dictcont,B=1)

dictaux = getDictionary(auxiliary)
print('  Auxiliary Dictionary = Vector 1 x',len(dictaux))
printme(A=dictaux,B=1)

print('3.Finding Term Frequency (TF)',end='')
TF=[];temp=[];count=0
for i in content:
    for j in dictcont:
        for k in i:
            if j==k:
                count=count+1
        temp.append(count)
        count=0
    TF.append(temp)
    temp=[]
print('...Done')
print('  TF = Matrix',len(TF),'x',len(TF[0]))
printme(A=TF,B=len(TF))

print('4.Finding Document Fequency(DF)',end='')
Freq=[];count=0
for i in range(0,len(TF[0])):
    for j in range(0,len(TF)):
        if TF[j][i]>0:
            count=count+1
    Freq.append(count)
    count=0
print('...Done')
print('  DF = Vector 1 x',len(Freq))
printme(A=Freq,B=1)

print('5.Finding Inverse Document Frequency (IDF)',end='')
IDF=[]
for i in Freq:
    x = math.log10(len(content)/i)
    IDF.append(x)
print('...Done')
print('  IDF = Vector 1 x',len(IDF))
printme(A=IDF,B=1)

print('6.Finding Weight(TF*IDF)',end='')
TFIDF=[];temp=[]
for i in range(0,len(TF)):
    for j in range(0,len(TF[i])):
        temp.append(TF[i][j]*IDF[j])
    TFIDF.append(temp)
    temp=[]
print('...Done')
print('  TFIDF = Matrix',len(TFIDF),'x',len(TFIDF[0]))
printme(A=TFIDF,B=len(TFIDF))

hold = input('Input any value to continue...')

print('\n=====================================')
print('Content Based Algorithm Using K-Means')
print('=====================================')
Iterasi=1
print('ITERATION ',Iterasi)

K = 3
print('K =',K)

print('1.Initialize Random Centroid',end='')
centroid=[]
temp = [0,440,25]
for i in range(0,K):
    centroid.append(TFIDF[temp[i]-1])
print('...Done')

print('2.Compute Euclidean Distance',end='')
Euclid = EuclideanDistance(centroid,TFIDF)
print('...Done')

print('3.Assigned Document to Closest Centroid',end='')
indeksKmeans1 = getIndeks(0,Euclid,TFIDF)
print('...Done')
            
ulang=True
Iterasi=Iterasi+1
while(ulang==True):
    print('\nITERATION ',Iterasi)
    
    print('1.Finding New Centroid')
    centroid1 = getNewCentroid(indeksKmeans1,TFIDF,K)

    print('2.Compute Euclidian Distance Document to New Centroid',end='')
    Euclid = EuclideanDistance(centroid1,TFIDF)
    print('...Done')

    print('3.Assigned Document to New Closest Centroid',end='')
    print('  Done!')
    indeksKmeans2 = getIndeks (0,Euclid,TFIDF)

    if indeksKmeans1 == indeksKmeans2:
        print('\nAnggota Cluster Tidak Berubah')
        print('Proses Berhenti')
        printCluster(indeksKmeans2,K)
        ulang=False
    else:
        print('Anggota Cluster Berubah')
        print('Repeat the Process from Step 2')
        print('Centroid = New Centroid')
        print('')
        centroid = centroid1
        Iterasi = Iterasi+1

dictclust=[]
for i in range(0,K):
    temp=[]
    for j in range(0,len(indeksKmeans2)):
        if indeksKmeans2[j]==i:
            for k in content[j]:
                if k not in temp:
                    temp.append(k)
    dictclust.append(temp)

print('=====================')
print('First Minor Iteration')
print('=====================')
print('1.Finding Euclidean Distance',end='')
Euclid = EuclideanDistance(centroid1,TFIDF)
print('...Done!')

print('2.Assign Document to Closest Cluster',end='')
indeksFirstIteration = getIndeks(0,Euclid,TFIDF)
print('...Done!')

print('3.Update Cluster Centroid Fisrt Minor Iteration')
centroidFirstIteration = getNewCentroid(indeksFirstIteration,TFIDF,K)
printCluster(indeksFirstIteration,K)

hold = input('Input any value to continue...')

print('======================')
print('Second Minor Iteration')
print('======================')

print('1.Compute Giny Index',end='')
Frj=[];Frm=[];temp=[]
for i in range(0,len(auxiliary)):
    for j in range(0,K):
        count=0
        for k in auxiliary[i]:
            if k not in dictclust[j]:
                count=count+1
        temp.append(count)
        count=0
    Frj.append(temp)
    Frm.append(sum(Frj[i]))
    temp=[]

Prj=[];temp=[]
for i in range(0,len(Frj)):
    for j in range(0,len(Frj[0])):
        temp.append(Frj[i][j]/Frm[i])
    Prj.append(temp)
    temp=[]

Giny=[];temp=[];temp2=[]
for i in range(0,len(Prj)):
    for j in range(0,K):
        temp.append(math.pow(Prj[i][j],2))
    Giny.append(sum(temp))
    temp2.append(temp)
    temp=[]
print('...Done!')

print('2.Compute Average of Giny Index',end='')
temp=0
for i in range(0,len(Giny)):
    temp=temp+Giny[i]
Avg = temp/len(Giny)
print('...Done!')

print('3.Mark Attribute as Discrimanatory',end='')
Disc=[];idxdisc=[];Ri=[]
for i in range(0,len(Giny)):
    if Giny[i]<=Avg:
        Disc.append(content[i])
        Ri.append(i)
print('...Done!')

print('4.Compute Probability of Discriminatory Attribute',end='')
count=0;temp=[];temp2=[]
for i in range(len(Ri)):
    for j in range(0,K):
        for k in auxiliary[Ri[i]]:
            if k in dictclust[j]:
                count=count+1
        temp.append(count)
        count=0
    temp2.append(temp)
    temp=[]
print('...Done!')

print('5.Assigned Discriminatory Attribute to Cluster',end='')
idx=[]
for i in range(0,len(indeksFirstIteration)):
    idx.append(indeksFirstIteration[i])
    
indeksSecondIteration=idx
for j in range(0,len(temp2)):
    x = max(temp2[j]) #identify if discriminatory attribute assigned to other cluster
    idxx = temp2[j].index(x)
    indeksSecondIteration[Ri[j]]=idxx
print('...Done!')

print('6.Update Cluster Centroid')
centroidSecondIteration = getNewCentroid(indeksSecondIteration,TFIDF,K)
printCluster(indeksSecondIteration,K)

for i in range(0,len(content)):
    if indeksFirstIteration[i] != indeksSecondIteration[i]:
        print('Doc',i+1,'Move from Cluster',indeksFirstIteration[i]+1,'to Cluster',indeksSecondIteration[i]+1)
