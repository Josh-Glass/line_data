import pandas as pd
from sklearn.manifold import MDS 
import numpy as np
import matplotlib.pyplot as plt

def getSims(path):
    df = pd.read_csv(path)
    group= df.groupby(['stimId1','stimID2'], as_index= False)['response'].mean()
    #print(group)
   # group.to_csv('cond_1_grouped_sims.csv')
   #make a bunch of lists to restructure the dataframe
    stimlist = [] #this list will be the index column of items
    #these lists will hold similarity data between the item in the stimlist and the item named in the list label

    #scale the data
    group['response']= group['response']/(6-0)

    A50=np.array(group[group['stimID2'] == 'A50']['response'])
    A150=np.array(group[group['stimID2'] == 'A150']['response'])
    A250=np.array(group[group['stimID2'] == 'A250']['response'])
    A350=np.array(group[group['stimID2'] == 'A350']['response'])
    A450=np.array(group[group['stimID2'] == 'A450']['response'])
    B550=np.array(group[group['stimID2'] == 'B550']['response'])
    B650=np.array(group[group['stimID2'] == 'B650']['response'])
    B750=np.array(group[group['stimID2'] == 'B750']['response'])
    B850=np.array(group[group['stimID2'] == 'B850']['response'])
    B950=np.array(group[group['stimID2'] == 'B950']['response'])
    C1050=np.array(group[group['stimID2'] == 'C1050']['response'])
    C1250=np.array(group[group['stimID2'] == 'C1250']['response'])
    C1350=np.array(group[group['stimID2'] == 'C1350']['response'])
    C1450=np.array(group[group['stimID2'] == 'C1450']['response'])
    C1150=np.array(group[group['stimID2'] == 'C1150']['response'])


    


   




    for item in group['stimId1'].unique():
        stimlist.append(item)
        #break
    #print(stimlist)

    df = pd.DataFrame({
        'stim': stimlist[0:-1],
        'A50':A50, 
        'A150':A150,
        'A250':A250, 
        'A350':A350,
        'A450':A450,
        'B550':B550, 
        'B650':B650,
        'B750':B750,
        'B850':B850,
        'B950':B950,
        'C1050':C1050,
        'C1150':C1150,
        'C1250':C1250,
        'C1350':C1350,
        'C1450':C1450

    })
    df.to_csv('MDS/mds_cond1.csv')


path = 'C:/Users/apers/line_data/cond_1_similarity_results.csv'   
#getSims(path=path)



def Do_mds(path):
    df= pd.read_csv(path, index_col='stim')
    df = (df/(0-1))+1 #convert from similarity scores to dissimilarity scores
    #print(df)
    mds= MDS(n_components=1,
              metric=True,
              dissimilarity='precomputed',
              #n_init=40,
              #max_iter=300,
              #verbose=0,
              #eps=0.001,
              #random_state=42,
              )
    df_scaled= mds.fit_transform(df)
    print(df_scaled)
    print(mds.stress_)

    y= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    c=['r','r','r','r','r',
       'b','b','b','b','b',
       'c','c','c','c','c']

    plt.scatter(x=df_scaled[0:5,0], y=y[0:5],color='r',s=500, marker=r'$\alpha$')
    plt.scatter(x=df_scaled[5:10,0], y=y[5:10],color='b',s=500, marker=r'$\beta$')
    plt.scatter(x=df_scaled[10:15,0], y=y[10:15],color='g',s=500, marker=r'$\gamma$')


    plt.show()




mdspath= 'C:/Users/apers/line_data/MDS/mds_cond1.csv'

Do_mds(path =mdspath)