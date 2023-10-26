import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plotLearnCurve(path, title):
    df = pd.read_csv(path)



    df['accuracy'] = df['accuracy'].astype(str)
    df = df.replace('True', 1)
    df = df.replace('False', 0)

    means_df = df.groupby(['block'], as_index= True)['accuracy'].describe()
    
    means_df.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    block= np.array(df['block'].unique()) + 1
    means = np.array(means_df['means'])
    numSubs= len(df['id'].unique())
    sem = np.array(means_df['stndDev']/np.sqrt(numSubs))

    plt.errorbar(block, means, yerr= sem, color = 'k', marker= 'o', markersize= 10)
    
    plt.ylim(0.2,1.05)
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    xlabels = [1,2,3,4,5]
    plt.xticks(xlabels)
    plt.xlabel("Training Block")
    plt.ylabel("Proportioin Correct")
    plt.title(str(title)+"(n="+str(numSubs)+")")
    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()







def plotTyps(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])

    numSubs= len(df['id'].unique())


    

    means_df = df.groupby(['stimId'], as_index= True)['response'].describe()
    
    means_df.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    means = np.array(means_df['means'])
    sem = np.array(means_df['stndDev']/np.sqrt(numSubs))

    regions = np.array(means_df['means'].index)
    x_pos = np.arange(len(regions))

    # Build the plot
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax.bar(x_pos[0:5], means[0:5], yerr=sem[0:5], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '.')
    ax.bar(x_pos[5:10], means[5:10], yerr=sem[5:10], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '/')
    ax.bar(x_pos[10:15], means[10:15], yerr=sem[10:15], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= 'o')
    
    ax.set_ylabel('Typicalty')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(title)+"(n="+str(numSubs)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()


    



def threeGetGens(path, title):
    df = pd.read_csv(str(path))
    newdf = df.drop(index = df[df['category'] != 'test'].index)
    group= newdf.groupby(['stimId','response'], as_index= True)['id'].describe()
    group.rename(columns = {'count':'counts'}, inplace = True)
    
    print(title)
    print(group)

    counts= np.array(group['counts'])
    numsubs = len(np.array(df['id'].unique()))
    data = counts
    
    
    T1000= np.array(group['counts'].index[0][0]) #T1000
    A1000= np.array(group['counts'].index[0][1]) #TAlpha
    B1000=np.array(group['counts'].index[1][1]) #TBeta
    G1000= np.array(group['counts'].index[2][1]) #TGamma



    T500= np.array(group['counts'].index[2][0]) #T500
    A500= np.array(group['counts'].index[3][1]) #TAlpha
    B500= np.array(group['counts'].index[4][1]) #TBeta

    G500= np.array(group['counts'].index[5][1]) #TGamma




    regions = np.array([A1000, B1000,G1000])
    print(regions,"regions")
    x_pos = np.arange(len(regions))
    regions2 = np.array([A500,B500,G500])
    print(regions2, "regions2")

    x_pos2 = np.arange(len(regions2))

    # Build the plot
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax.bar(x_pos[0], data[0], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='r', capsize=4, width=0.7)
    ax.bar(x_pos[1], data[1], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='b', capsize=4, width=0.7)
    ax.bar(x_pos[2], data[2], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='g', capsize=4, width=0.7)
    
    ax.set_ylabel('Count of Subjects')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(T1000)+'\n'+str(title)+"(n="+str(numsubs)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(T1000)  +' '+ str(title)+'.png')
    #plt.show()
    plt.clf()

    fig1, ax1 = plt.subplots()

    ax1.bar(x_pos2[0], data[3], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='r', capsize=4, width=0.7)
    ax1.bar(x_pos2[1], data[4], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='b', capsize=4, width=0.7)
    ax1.bar(x_pos2[2], data[5], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='g', capsize=4, width=0.7)
    
    ax1.set_ylabel('Count of Subjects')
    ax1.set_xticks(x_pos2)
    ax1.set_xticklabels(regions2)
    ax1.set_title('T500'+'\n'+str(title)+"(n="+str(numsubs)+")")
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+'T500' +' '+ str(title)+'.png')
    #plt.show()
    plt.clf()










def twoGetGens(path, title):
    df = pd.read_csv(str(path))
    newdf = df.drop(index = df[df['category'] != 'test'].index)
    group= newdf.groupby(['stimId','response'], as_index= True)['id'].describe()
    group.rename(columns = {'count':'counts'}, inplace = True)
    


    print(title)
    print(group)

    counts= np.array(group['counts'])
    numsubs = len(np.array(df['id'].unique()))
    data = counts
    
    
    T1000= np.array(group['counts'].index[0][0]) #T1000
    B1000=np.array(group['counts'].index[0][1]) #TBeta
    G1000= np.array(group['counts'].index[1][1]) #TGamma



    T500= np.array(group['counts'].index[2][0]) #T500
    A500= np.array(group['counts'].index[2][1]) #TAlpha
    B500= np.array(group['counts'].index[3][1]) #TBeta





    regions = np.array([B1000, G1000])
    print(regions,"regions")
    x_pos = np.arange(len(regions))
    regions2 = np.array([A500,B500])
    print(regions2, "regions2")

    x_pos2 = np.arange(len(regions2))

    # Build the plot
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax.bar(x_pos[0], data[0], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7)
    ax.bar(x_pos[1], data[1], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7)
    
    ax.set_ylabel('Count of Subjects')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(T1000)+'\n'+str(title)+"(n="+str(numsubs)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(T1000)  +' '+ str(title)+'.png')
    #plt.show()
    plt.clf()

    fig1, ax1 = plt.subplots()

    ax1.bar(x_pos2[0], data[2], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='b', capsize=4, width=0.7)
    ax1.bar(x_pos2[1], data[3], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='g', capsize=4, width=0.7)
    
    ax1.set_ylabel('Count of Subjects')
    ax1.set_xticks(x_pos2)
    ax1.set_xticklabels(regions2)
    ax1.set_title('T500'+'\n'+str(title)+"(n="+str(numsubs)+")")
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+'T500' +' '+ str(title)+'.png')
    #plt.show()
    plt.clf()






'''

path = 'C:/Users/apers/line_data/cond_1_testPhase_results.csv'
title= 'Observe A - Classify B & C'
getgens(path=path, title=title)'''






'''
def getSims(path):
    df = pd.read_csv(path)
    group= df.groupby(['stimId1','stimID2'], as_index= True)['response'].describe()
    print(group)
    group.to_csv('cond_1_grouped_sims.csv')

path = 'C:/Users/apers/line_data/cond_1_similarity_results.csv'   
getSims(path=path)'''

