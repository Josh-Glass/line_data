import pandas as pd
from sklearn.manifold import MDS 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import euclidean_distances

def getSims(path, title):
    df = pd.read_csv(path)
    df= df.replace('A50', 'A050')
    group= df.groupby(['stimId1','stimID2'], as_index= False)['response'].mean()
    #print(group)
   # group.to_csv('cond_1_grouped_sims.csv')
   #make a bunch of lists to restructure the dataframe
    stimlist = ['A50', 'A150', 'A250', 'A350', 'A450', 'B550', 'B650', 'B750', 'B850', 'B950', 'C1050', 'C1150', 'C1250', 'C1350', 'C1450'] #this list will be the index column of items
    #these lists will hold similarity data between the item in the stimlist and the item named in the list label

    #scale the data
    group['response']= group['response']/(6-0)
    #group.to_csv('MDS/groupedSimRatings.csv')

    A50L1=np.array(group[group['stimId1'] == 'A050']['response'])
    A50L2=np.array(group[group['stimID2'] == 'A050']['response'])
    A50 = np.mean([A50L1, A50L2], axis=0)
    A50= np.insert(A50, 0, 1)
    print(np.array(group[group['stimId1'] == 'A050']['stimID2']))
    print(np.array(group[group['stimID2'] == 'A050']['stimId1']))


    A150L1=np.array(group[group['stimId1'] == 'A150']['response'])
    A150L2=np.array(group[group['stimID2'] == 'A150']['response'])
    A150 = np.mean([A150L1, A150L2], axis=0)
    A150= np.insert(A150, 1, 1)
    print(np.array(group[group['stimId1'] == 'A150']['stimID2']))
    print(np.array(group[group['stimID2'] == 'A150']['stimId1']))


    A250L1=np.array(group[group['stimId1'] == 'A250']['response'])
    A250L2=np.array(group[group['stimID2'] == 'A250']['response'])
    A250 = np.mean([A250L1, A250L2], axis=0)
    A250= np.insert(A250, 2, 1)


    A350L1=np.array(group[group['stimId1'] == 'A350']['response'])
    A350L2=np.array(group[group['stimID2'] == 'A350']['response'])
    A350 = np.mean([A350L1, A350L2], axis=0)
    A350= np.insert(A350, 3, 1)



    A450L1=np.array(group[group['stimId1'] == 'A450']['response'])
    A450L2=np.array(group[group['stimID2'] == 'A450']['response'])
    A450 = np.mean([A450L1, A450L2], axis=0)
    A450= np.insert(A450, 4, 1)


    B550L1=np.array(group[group['stimId1'] == 'B550']['response'])
    B550L2=np.array(group[group['stimID2'] == 'B550']['response'])
    B550 = np.mean([B550L1, B550L2], axis=0)
    B550 = np.insert(B550, 5, 1)


    B650L1=np.array(group[group['stimId1'] == 'B650']['response'])
    B650L2=np.array(group[group['stimID2'] == 'B650']['response'])
    B650 = np.mean([B650L1, B650L2], axis=0)
    B650 = np.insert(B650, 6, 1)

    B750L1=np.array(group[group['stimId1'] == 'B750']['response'])
    B750L2=np.array(group[group['stimID2'] == 'B750']['response'])
    B750 = np.mean([B750L1, B750L2], axis=0)
    B750 = np.insert(B750, 7, 1)



    B850L1=np.array(group[group['stimId1'] == 'B850']['response'])
    B850L2=np.array(group[group['stimID2'] == 'B850']['response'])
    B850 = np.mean([B850L1, B850L2], axis=0)
    B850 = np.insert(B850, 8, 1)



    B950L1=np.array(group[group['stimId1'] == 'B950']['response'])
    B950L2=np.array(group[group['stimID2'] == 'B950']['response'])
    B950 = np.mean([B950L1, B950L2], axis=0)
    B950 = np.insert(B950, 9, 1)

    C1050L1=np.array(group[group['stimId1'] == 'C1050']['response'])
    C1050L2=np.array(group[group['stimID2'] == 'C1050']['response'])
    C1050 = np.mean([C1050L1, C1050L2], axis=0)
    C1050 = np.insert(C1050, 10, 1)

    C1150L1=np.array(group[group['stimId1'] == 'C1150']['response'])
    C1150L2=np.array(group[group['stimID2'] == 'C1150']['response'])
    C1150 = np.mean([C1150L1, C1150L2], axis=0)
    C1150 = np.insert(C1150, 11, 1)


    C1250L1=np.array(group[group['stimId1'] == 'C1250']['response'])
    C1250L2=np.array(group[group['stimID2'] == 'C1250']['response'])
    C1250 = np.mean([C1250L1, C1250L2], axis=0)
    C1250 = np.insert(C1250, 12, 1)




    C1350L1=np.array(group[group['stimId1'] == 'C1350']['response'])
    C1350L2=np.array(group[group['stimID2'] == 'C1350']['response'])
    C1350 = np.mean([C1350L1, C1350L2], axis=0)
    C1350 = np.insert(C1350, 13, 1)


    C1450L1=np.array(group[group['stimId1'] == 'C1450']['response'])
    C1450L2=np.array(group[group['stimID2'] == 'C1450']['response'])
    C1450 = np.mean([C1450L1, C1450L2], axis=0)
    C1450 = np.insert(C1450, 14, 1)

    print(len(A50))
    print(len(A150))
    print(len(A250))
    print(len(A350))
    print(len(A450))

    print(len(B550))
    print(len(B650))
    print(len(B750))
    print(len(B850))
    print(len(B950))

    print(len(C1050))
    print(len(C1150))
    print(len(C1250))
    print(len(C1350))
    print(len(C1450))



   




    #for item in group['stimId1'].unique():
     #   stimlist.append(item)
        #break
    #print(stimlist)

    df = pd.DataFrame({
        'stim': stimlist,
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
    df.to_csv('MDS/' + str(title) + '.csv', index= None)


path = 'C:/Users/apers/line_data/cond_3_similarity_results.csv'
title= "NoObservationSim"   
#getSims(path=path, title = title)























def Do_mds(path, title):
    df= pd.read_csv(path, index_col='stim')
    df = (df/(0-1))+1 #convert from similarity scores to dissimilarity scores
    #print(df)
    mds= MDS(n_components=1,
              metric=True,
              dissimilarity='precomputed',
              n_init=500,
              max_iter=3000,
              #verbose=0,
              eps=0.0001,
              random_state=3,
              )
    df_scaled= mds.fit_transform(df)
    #print(df_scaled)
    #print(mds.stress_)

    points = mds.embedding_
    #print(points)
    DE = euclidean_distances(points)
    stress = mds.stress_ / (np.sum((DE - np.mean(df.values))**2))
    print(stress)

    y= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    c=['r','r','r','r','r',
       'b','b','b','b','b',
       'c','c','c','c','c']
    xT500=  df_scaled[4,0] + ((500 - 450) / (550 - 450))*(df_scaled[5,0] - df_scaled[4,0])
    #yT500=  df_scaled[4,1] + ((500 - 450) / (550 - 450))*(df_scaled[5,1] - df_scaled[4,1])

    xT1000=  df_scaled[9,0] + ((1000 - 950) / (1050 - 950))*(df_scaled[10,0] - df_scaled[9,0])
    #yT1000=  df_scaled[9,1] + ((1000 - 950) / (1050 - 950))*(df_scaled[10,1] - df_scaled[9,1])
    ax= plt.subplot()
    ax.scatter(x=df_scaled[0:5,0], y=y[0:5], color='r',s=500, marker=r'$\alpha$')
    ax.scatter(x=df_scaled[5:10,0], y=y[5:10],color='b',s=500, marker=r'$\beta$')
    ax.scatter(x=df_scaled[10:15,0], y=y[10:15],color='g',s=500, marker=r'$\gamma$')


    #ax.scatter(x=np.mean(df_scaled[0:5,0]), y=0,edgecolors='k',linewidths=2.5,color='None',s=500, marker='*')
    #ax.scatter(x=np.mean(df_scaled[5:10,0]), y=0,edgecolors='k',linewidths=2.5,color='None',s=500, marker='*')
    #ax.scatter(x=np.mean(df_scaled[10:15,0]), y=0,edgecolors='k',linewidths=2.5,color='None',s=500, marker='*')


    ax.scatter(x=xT500, y=0,color='k',s=500, marker=r'$T$')
    ax.scatter(x=xT1000, y=0,color='k',s=500, marker=r'$T$')


    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_major_locator(ticker.NullLocator())

    ax.set_title(str(title) + '\nNormalized Stress = ' + str(np.round(stress,decimals=3)))
    




    def cityblock_distance(A, B): 
    
        result = np.sum([abs(a - b) for (a, b) in zip(A, B)]) 
        return result


    #print('distance from t500-- beta - alpha', (cityblock_distance([xT500], df_scaled[5]) - cityblock_distance([xT500], df_scaled[4])))
    #print('distance from t1000-- gamma - beta', (cityblock_distance([xT1000], df_scaled[10]) - cityblock_distance([xT1000], df_scaled[9])))

    plt.show()




mdspath= 'C:/Users/apers/line_data/MDS/ReducedClassSim.csv'
title = ''
Do_mds(path =mdspath, title=title)