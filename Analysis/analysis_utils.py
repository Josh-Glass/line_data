import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import scikit_posthocs as skpost
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools
import matplotlib.colors as colors


# function to calculate Cohen's d for independent samples-- taken from https://machinelearningmastery.com/effect-size-measures-in-python/
def cohend(d1, d2):
 # calculate the size of samples
 n1, n2 = len(d1), len(d2)
 # calculate the variance of the samples
 s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
 # calculate the pooled standard deviation
 s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
 # calculate the means of the samples
 u1, u2 = np.mean(d1), np.mean(d2)
 # calculate the effect size
 return (u1 - u2) / s


def plotLearnCurve(path, title):
    df = pd.read_csv(path)
    


    df['accuracy'] = df['accuracy'].astype(str)
    df = df.replace('True', 1)
    df = df.replace('False', 0)

    



    




    #GET OVERALL LEARN CURVES
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








def fullPerCatLearnCurves(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])


    df['accuracy'] = df['accuracy'].astype(str)
    df = df.replace('True', 1)
    df = df.replace('False', 0)

    dfA = df.drop(index = df[df['category'] != 'Alpha'].index)
    dfB = df.drop(index = df[df['category'] != 'Beta'].index)
    dfG = df.drop(index = df[df['category'] != 'Gamma'].index)
    numSubs= len(df['id'].unique())
    block= np.array(df['block'].unique()) + 1




    #GET PERCAT LEARN CURVES (ALL THREE CATS ON ONE GRAPH)
    #alpha cat data
    means_dfA = dfA.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfA.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    meansA = np.array(means_dfA['means'])
    semA = np.array(means_dfA['stndDev']/np.sqrt(numSubs))

    #beta cat data
    means_dfB = dfB.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfB.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    meansB = np.array(means_dfB['means'])
    semB = np.array(means_dfB['stndDev']/np.sqrt(numSubs))


    #gamma cat
    means_dfG = dfG.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfG.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    meansG = np.array(means_dfG['means'])
    semG = np.array(means_dfG['stndDev']/np.sqrt(numSubs))

    #plot alpha line
    plt.errorbar(block, meansA, yerr= semA, color = 'r', marker= r'$\alpha$', markersize= 15,)
    #plot beta Line
    plt.errorbar(block, meansB, yerr= semB, color = 'b', marker= r'$\beta$', markersize= 15,)
    #plot gamma line
    plt.errorbar(block, meansG, yerr= semG, color = 'g', marker= r'$\gamma$', markersize= 15,)

    
    plt.ylim(0.2,1.05)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    xlabels = [1,2,3,4,5]
    plt.xticks(xlabels)
    plt.xlabel("Training Block")
    plt.ylabel("Proportioin Correct")
    plt.title(str(title)+"(n="+str(numSubs)+")")
    plt.savefig('Analysis/graphs/'+'perCat'+str(title) + '.png')
    plt.clf()


    #GET PER ITEM LEARN CUREVES (EACH CAT ON SEPARATE PLOT, I.E., FIVE CURVES PER PLOT)

    As= ['A1', 'A2', 'A3', 'A4', 'A5']
    marks= ['o', '*', 'D', 'v', 'X']
    Bs= ['B1', 'B2', 'B3', 'B4', 'B5']
    Gs= ['G1', 'G2', 'G3', 'G4', 'G5']


    for (i,e) in zip(As,marks):
        item_means_dfA = dfA[dfA['stimId']==str(i)].groupby(['block'], as_index= True)['accuracy'].describe()
        item_means_dfA.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
        item_meansA = np.array(item_means_dfA['means'])
        item_semA = np.array(item_means_dfA['stndDev']/np.sqrt(numSubs))
        plt.errorbar(block, item_meansA, yerr= item_semA, color = 'r', marker= e, markersize= 10,label=str(i))
        plt.ylim(0,1.05)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        xlabels = [1,2,3,4,5]
        plt.xticks(xlabels)
        plt.xlabel("Training Block")
        plt.ylabel("Proportioin Correct")
        plt.title('Alpha '+str(title)+"(n="+str(numSubs)+")")
    
    plt.savefig('Analysis/graphs/'+'perAItem'+str(title) + '.png')
    plt.clf()


    for (i,e) in zip(Bs,marks):
        item_means_dfB = dfB[dfB['stimId']==str(i)].groupby(['block'], as_index= True)['accuracy'].describe()
        item_means_dfB.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
        item_meansB = np.array(item_means_dfB['means'])
        item_semB = np.array(item_means_dfB['stndDev']/np.sqrt(numSubs))
        plt.errorbar(block, item_meansB, yerr= item_semB, color = 'b', marker= e, markersize= 10,label=str(i))
        plt.ylim(0,1.05)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        xlabels = [1,2,3,4,5]
        plt.xticks(xlabels)
        plt.xlabel("Training Block")
        plt.ylabel("Proportioin Correct")
        plt.title('Beta '+str(title)+"(n="+str(numSubs)+")")
    
    plt.savefig('Analysis/graphs/'+'perBItem'+str(title) + '.png')
    plt.clf()



    for (i,e) in zip(Gs,marks):
        item_means_dfG = dfG[dfG['stimId']==str(i)].groupby(['block'], as_index= True)['accuracy'].describe()
        item_means_dfG.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
        item_meansG = np.array(item_means_dfG['means'])
        item_semG = np.array(item_means_dfG['stndDev']/np.sqrt(numSubs))
        plt.errorbar(block, item_meansG, yerr= item_semG, color = 'g', marker= e, markersize= 10,label=str(i))
        plt.ylim(0,1.05)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        xlabels = [1,2,3,4,5]
        plt.xticks(xlabels)
        plt.xlabel("Training Block")
        plt.ylabel("Proportioin Correct")
        plt.title('Gamma '+str(title)+"(n="+str(numSubs)+")")
    
    plt.savefig('Analysis/graphs/'+'perGItem'+str(title) + '.png')
    plt.clf()






def reducedPerCatLearnCurves(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])


    df['accuracy'] = df['accuracy'].astype(str)
    df = df.replace('True', 1)
    df = df.replace('False', 0)

    dfB = df.drop(index = df[df['category'] != 'Beta'].index)
    dfG = df.drop(index = df[df['category'] != 'Gamma'].index)
    numSubs= len(df['id'].unique())
    block= np.array(df['block'].unique()) + 1




    #GET PERCAT LEARN CURVES (ALL THREE CATS ON ONE GRAPH)
   

    #beta cat data
    means_dfB = dfB.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfB.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    meansB = np.array(means_dfB['means'])
    semB = np.array(means_dfB['stndDev']/np.sqrt(numSubs))


    #gamma cat
    means_dfG = dfG.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfG.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    meansG = np.array(means_dfG['means'])
    semG = np.array(means_dfG['stndDev']/np.sqrt(numSubs))

    
    #plot beta Line
    plt.errorbar(block, meansB, yerr= semB, color = 'b', marker= r'$\beta$', markersize= 15,)
    #plot gamma line
    plt.errorbar(block, meansG, yerr= semG, color = 'g', marker= r'$\gamma$', markersize= 15,)

    
    plt.ylim(0.2,1.05)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    xlabels = [1,2,3,4,5]
    plt.xticks(xlabels)
    plt.xlabel("Training Block")
    plt.ylabel("Proportioin Correct")
    plt.title(str(title)+"(n="+str(numSubs)+")")
    plt.savefig('Analysis/graphs/'+'perCat'+str(title) + '.png')
    plt.clf()


    #GET PER ITEM LEARN CUREVES (EACH CAT ON SEPARATE PLOT, I.E., FIVE CURVES PER PLOT)

    marks= ['o', '*', 'D', 'v', 'X']
    Bs= ['B1', 'B2', 'B3', 'B4', 'B5']
    Gs= ['G1', 'G2', 'G3', 'G4', 'G5']




    for (i,e) in zip(Bs,marks):
        item_means_dfB = dfB[dfB['stimId']==str(i)].groupby(['block'], as_index= True)['accuracy'].describe()
        item_means_dfB.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
        item_meansB = np.array(item_means_dfB['means'])
        item_semB = np.array(item_means_dfB['stndDev']/np.sqrt(numSubs))
        plt.errorbar(block, item_meansB, yerr= item_semB, color = 'b', marker= e, markersize= 10,label=str(i))
        plt.ylim(0,1.05)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        xlabels = [1,2,3,4,5]
        plt.xticks(xlabels)
        plt.xlabel("Training Block")
        plt.ylabel("Proportioin Correct")
        plt.title('Beta '+str(title)+"(n="+str(numSubs)+")")
    
    plt.savefig('Analysis/graphs/'+'perBItem'+str(title) + '.png')
    plt.clf()



    for (i,e) in zip(Gs,marks):
        item_means_dfG = dfG[dfG['stimId']==str(i)].groupby(['block'], as_index= True)['accuracy'].describe()
        item_means_dfG.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
        item_meansG = np.array(item_means_dfG['means'])
        item_semG = np.array(item_means_dfG['stndDev']/np.sqrt(numSubs))
        plt.errorbar(block, item_meansG, yerr= item_semG, color = 'g', marker= e, markersize= 10,label=str(i))
        plt.ylim(0,1.05)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.65, top=0.85)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        xlabels = [1,2,3,4,5]
        plt.xticks(xlabels)
        plt.xlabel("Training Block")
        plt.ylabel("Proportioin Correct")
        plt.title('Gamma '+str(title)+"(n="+str(numSubs)+")")
    
    plt.savefig('Analysis/graphs/'+'perGItem'+str(title) + '.png')
    plt.clf()















def getTestAcc(path, title, changestimID, changes):
    df = pd.read_csv(path)
    df = df.replace(' Beta', 'Beta')

    for index, row in df.iterrows():
        #print(row['category'], row['response'], row['accuracy'], index)
        if row['response'] == row['category']:
            df.at[index, 'accuracy'] = 1

    df = df.replace('True', 1)
    df = df.replace('False', 0)
    df = df.drop(index = df[df['category'] == 'test'].index)
    numSubs= len(df['id'].unique())

    print(title)
    #print('Overall: ', df['accuracy'].mean(), df['accuracy'].std())
    #print('Alpha: ',df[df['category']=='Alpha']['accuracy'].mean(), ',', df[df['category']=='Alpha']['accuracy'].std())
    #print('Beta: ',df[df['category']=='Beta']['accuracy'].mean(), ',', df[df['category']=='Beta']['accuracy'].std())
    #print('Gamma: ',df[df['category']=='Gamma']['accuracy'].mean(), ',', df[df['category']=='Gamma']['accuracy'].std())
    #print(f_oneway(df[df['category']=='Alpha']['accuracy'],df[df['category']=='Beta']['accuracy'],df[df['category']=='Gamma']['accuracy']))
    #print(skpost.posthoc_scheffe(a=[df[df['category']=='Alpha']['accuracy'],df[df['category']=='Beta']['accuracy'],df[df['category']=='Gamma']['accuracy']]))
    print('              ')
    catnames = ['Alpha', 'Beta', 'Gamma']
    cats = np.arange(len(catnames))

    fig0, ax0 = plt.subplots()
    ax0.bar(cats[0], df[df['category']=='Alpha']['accuracy'].mean(), yerr=df[df['category']=='Alpha']['accuracy'].sem(), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '.')
    ax0.bar(cats[1], df[df['category']=='Beta']['accuracy'].mean(), yerr=df[df['category']=='Beta']['accuracy'].sem(), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '/')
    ax0.bar(cats[2], df[df['category']=='Gamma']['accuracy'].mean(), yerr=df[df['category']=='Gamma']['accuracy'].sem(), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= 'o')
    ax0.set_ylim(0,1)
    ax0.set_ylabel('Proportion Correct')
    ax0.set_xticks(cats)
    ax0.set_xticklabels(catnames)
    ax0.set_title(str(title)+"(n="+str(numSubs)+")")
    ax0.yaxis.grid(False)
    ax0.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+'percat'+str(title) + '.png')
    plt.clf()




    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])



    

    means = np.array(df.groupby(['stimId'], as_index= True)['accuracy'].mean())
    #print(means)
    sem = np.array(df.groupby(['stimId'], as_index= True)['accuracy'].sem())

    
    

    regions = np.array(df.groupby(['stimId'], as_index= True)['accuracy'].mean().index)
    x_pos = np.arange(len(regions))

    # Build the plot
    fig, ax = plt.subplots()
    

    ax.bar(x_pos[0:5], means[0:5], yerr=sem[0:5], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '.')
    ax.bar(x_pos[5:10], means[5:10], yerr=sem[5:10], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '/')
    ax.bar(x_pos[10:15], means[10:15], yerr=sem[10:15], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= 'o')
    ax.set_ylim(0,1)
    ax.set_ylabel('Proportion Correct')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(title)+"(n="+str(numSubs)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()
    #plt.show()







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
    ax.set_ylim(0,6)
    ax.set_ylabel('Typicality')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(title)+"(n="+str(numSubs)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()


    


def individualTyps(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])

    numSubs= len(df['id'].unique())


    
    for subject in df['id'].unique():
        means_df = df[df['id']== subject].groupby(['stimId'], as_index= True)['response'].mean()

        print(subject)
        print(np.array(means_df.index))
        means = np.array(means_df)
        regions = np.array(means_df.index)
        x_pos = np.arange(len(regions))
    
        # Build the plot
        fig, ax = plt.subplots()
        #plt.figure(figsize=(20, 3))
        #plt.ylim(0,1)
        ax.bar(x_pos[0:5], means[0:5], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '.')
        ax.bar(x_pos[5:10], means[5:10], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '/')
        ax.bar(x_pos[10:15], means[10:15], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= 'o')
        ax.set_ylim(0,7)
        ax.set_ylabel('Typicality')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions)
        ax.set_title(str(title)+" subject "+str(subject))
        ax.yaxis.grid(False)
        ax.tick_params(axis='x', which='major', labelsize=10)

        plt.savefig('Analysis/graphs/individual_typs/'+'ss'+str(subject)+'_'+str(title) + '.png')
        plt.clf()


























def logPlotTyp2(path, title, changestimID, changes):
    df = pd.read_csv(path)
    df = df.replace(' Beta', 'Beta')

    #change the stim ID to pixel value of the line
    if changestimID == True:
        for i in changes:
            df= df.replace(i, int(changes[i])) #needs to be an int

    numSubs= len(df['id'].unique())

    dfA = df.drop(index = df[df['category'] != 'Alpha'].index)
    dfB = df.drop(index = df[df['category'] != 'Beta'].index)
    dfG = df.drop(index = df[df['category'] != 'Gamma'].index)


    Ax = np.log10(np.array(dfA['stimId']))
    Ay = np.array(dfA['response']) +1


    Bx = np.log10(np.array(dfB['stimId']))
    By = np.array(dfB['response']) +1

    Gx = np.log10(np.array(dfG['stimId']))
    Gy = np.array(dfG['response']) +1

    #Alpha regression Line
    YA= Ay        
    XA = Ax
    XA = XA
    XA = sm.add_constant(XA)
    modelA= sm.OLS(YA,XA)
    resultsA = modelA.fit()
    Aregx= np.linspace(np.min(Ax), np.max(Ax), 1000)
    Aregy= resultsA.params[1]*Aregx + resultsA.params[0]
    plt.plot(Aregx, Aregy, c='red', linewidth=3.5)


    #Beta regression Line
    YB= By        
    XB = Bx
    XB = XB
    XB = sm.add_constant(XB)
    modelB= sm.OLS(YB,XB)
    resultsB = modelB.fit()
    Bregx= np.linspace(np.min(Bx), np.max(Bx), 1000)
    Bregy= resultsB.params[1]*Bregx + resultsB.params[0]
    plt.plot(Bregx, Bregy, c='blue', linewidth=3.5)


    #Gamma regression Line
    YG= Gy        
    XG = Gx
    XG = XG
    XG = sm.add_constant(XG)
    modelG= sm.OLS(YG,XG)
    resultsG = modelG.fit()
    Gregx= np.linspace(np.min(Gx), np.max(Gx), 1000)
    Gregy= resultsG.params[1]*Gregx + resultsG.params[0]
    plt.plot(Gregx, Gregy, c='green', linewidth=3.5)

    plt.scatter(x=Ax, y=Ay, marker='o', s=100, ec=colors.to_rgba('black',1), fc=colors.to_rgba('red',0.1), label=r'$\alpha$')
    plt.scatter(x=Bx, y=By, marker='s',s=100, ec=colors.to_rgba('black',1), fc=colors.to_rgba('blue',0.1), label=r'$\beta$')
    plt.scatter(x=Gx, y=Gy, marker='^', s=100, ec=colors.to_rgba('black',1), fc=colors.to_rgba('green',0.1), label=r'$\gamma$')
    plt.ylabel('Typicality Rating')
    plt.xlabel('log(line length)')
    plt.title(str(title)+'\nn='+ str(numSubs))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.80, top=0.85)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    #plt.show()
    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()














def logPlotTyp1(path, title, changestimID, changes):
    df = pd.read_csv(path)
    df = df.replace(' Beta', 'Beta')

    #change the stim ID to pixel value of the line
    if changestimID == True:
        for i in changes:
            df= df.replace(i, int(changes[i])) #needs to be an int

    numSubs= len(df['id'].unique())

    dfA = df.drop(index = df[df['category'] != 'Alpha'].index)


    Ax = np.log10(np.array(dfA['stimId']))
    Ay = np.array(dfA['response']) +1



    #Alpha regression Line
    YA= Ay        
    XA = Ax
    XA = XA
    XA = sm.add_constant(XA)
    modelA= sm.OLS(YA,XA)
    resultsA = modelA.fit()
    Aregx= np.linspace(np.min(Ax), np.max(Ax), 1000)
    Aregy= resultsA.params[1]*Aregx + resultsA.params[0]
    plt.plot(Aregx, Aregy, c='red', linewidth=3.5)



    plt.scatter(x=Ax, y=Ay, marker='o', s=100, ec=colors.to_rgba('black',1), fc=colors.to_rgba('red',0.1), label=r'$\alpha$')
    plt.ylabel('Typicality Rating')
    plt.xlabel('log(line length)')
    plt.title(str(title)+'\nn='+ str(numSubs))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.80, top=0.85)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    #plt.show()
    plt.savefig('Analysis/graphs/'+str(title) + '.png')
    plt.clf()




changes= {
    'A50': '50',
    'A150': '150',
    'A250': '250',
    'A350': '350',
    'A450': '450',

    'B550': '550',
    'B650': '650',
    'B750': '750',
    'B850': '850',
    'B950': '950',

    'C1050': '1050',
    'C1150': '1150',
    'C1250': '1250',
    'C1350': '1350',
    'C1450': '1450',
}


#logPlotTyp2(path='cond_2_typicality2_results.csv', title='Post-Classification Typicality- Full Classification', changestimID=True, changes=changes)














def threeGetGens(path, title):
    df = pd.read_csv(str(path))
    newdf = df.drop(index = df[df['category'] != 'test'].index)
    group= newdf.groupby(['stimId','response'], as_index= True)['id'].describe()
    group.rename(columns = {'count':'counts'}, inplace = True)
    
   

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
    x_pos = np.arange(len(regions))
    regions2 = np.array([A500,B500,G500])

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















def HardTestRight_GetGens(pathtest, pathlearn, title):
    df = pd.read_csv(str(pathtest))
    dflearn = pd.read_csv(str(pathlearn))

    dflearn = dflearn.drop(index = dflearn[dflearn['block'] != 4].index)

    dflearnA = dflearn.drop(index = dflearn[dflearn['stimId'] != 'A450'].index)
    dflearnFailA = dflearnA.drop(index = dflearnA[dflearnA['response'] == 'Alpha'].index)
    learnFailA = list(dflearnFailA['id'])
    print(learnFailA)
    print('')
    dflearnB = dflearn.drop(index = dflearn[dflearn['stimId'] != 'B950'].index)
    dflearnFailB = dflearnB.drop(index = dflearnB[dflearnB['response'] != 'Beta'].index)
    learnFailB = list(dflearnFailB['id'])
    print(learnFailB)


    df = df.replace(' Beta', 'Beta')

    for index, row in df.iterrows():
        if row['response'] == row['category']:
            df.at[index, 'accuracy'] = 1

    dfFailedAPnums = df[df['stimId']=='A450']
    dfFailedAPnums = dfFailedAPnums.drop(index = dfFailedAPnums[dfFailedAPnums['response'] == 'Alpha'].index)
    failedAPnums= list(dfFailedAPnums['id']) + learnFailA 
    #print('lista ', failedAPnums)

    dfA = df
    for item in failedAPnums:
        dfA = dfA.drop(index = dfA[dfA['id'] == item].index)


    dfFailedBPnums = df[df['stimId']=='B950']
    dfFailedBPnums = dfFailedBPnums.drop(index = dfFailedBPnums[dfFailedBPnums['response'] == 'Beta'].index)
    failedBPnums= list(dfFailedBPnums['id']) + learnFailB 
    #print('list b', failedBPnums)

    dfB = df
    for item in failedBPnums:
        dfB = dfB.drop(index = dfB[dfB['id'] == item].index)


    newdfB = dfB.drop(index = dfB[dfB['category'] != 'test'].index)
    groupB= newdfB.groupby(['stimId','response'], as_index= True)['id'].describe()
    groupB.rename(columns = {'count':'counts'}, inplace = True)
    #print(groupB)


    newdfA = dfA.drop(index = dfA[dfA['category'] != 'test'].index)
    groupA= newdfA.groupby(['stimId','response'], as_index= True)['id'].describe()
    groupA.rename(columns = {'count':'counts'}, inplace = True)
    #print(groupA)
   

    countsB= np.array(groupB['counts'])
    numsubsB = len(np.array(dfB['id'].unique()))
    dataB = countsB


    countsA= np.array(groupA['counts'])
    numsubsA = len(np.array(dfA['id'].unique()))
    dataA = countsA
    
    
    T1000= np.array(groupB['counts'].index[0][0]) #T1000
    B1000= np.array(groupB['counts'].index[0][1]) #TBeta
    G1000=np.array(groupB['counts'].index[1][1]) #TGamma



    T500= np.array(groupA['counts'].index[1][0]) #T500
    A500= np.array(groupA['counts'].index[2][1]) #TAlpha
    B500= np.array(groupA['counts'].index[3][1]) #TBeta





    regions = np.array([B1000,G1000])
    x_pos = np.arange(len(regions))
    regions2 = np.array([A500,B500])

    x_pos2 = np.arange(len(regions2))

    # Build the plot
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax.bar(x_pos[0], dataB[0], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='r', capsize=4, width=0.7)
    ax.bar(x_pos[1], dataB[1], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='b', capsize=4, width=0.7)
    
    ax.set_ylabel('Count of Subjects')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions)
    ax.set_title(str(T1000)+'\n'+'B5 Correct '+str(title)+"(n="+str(numsubsB)+")")
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+str(T1000)  +' B5Corr '+ str(title)+'.png')
    #plt.show()
    plt.clf()

    fig1, ax1 = plt.subplots()

    ax1.bar(x_pos2[0], dataA[2], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='r', capsize=4, width=0.7)
    ax1.bar(x_pos2[1], dataA[3], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='b', capsize=4, width=0.7)
    
    ax1.set_ylabel('Count of Subjects')
    ax1.set_xticks(x_pos2)
    ax1.set_xticklabels(regions2)
    ax1.set_title('T500'+'\n'+'A5 Correct '+str(title)+"(n="+str(numsubsA)+")")
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('Analysis/graphs/'+'T500' +' A5Corr '+ str(title)+'.png')
    #plt.show()
    plt.clf()


#HardTestRight_GetGens(pathlearn='cond_3_trainPhase_results.csv', pathtest='cond_3_testPhase_results.csv',title='at Train And Test (No Observation)')









def twoGetGens(path, title):
    df = pd.read_csv(str(path))
    newdf = df.drop(index = df[df['category'] != 'test'].index)
    group= newdf.groupby(['stimId','response'], as_index= True)['id'].describe()
    group.rename(columns = {'count':'counts'}, inplace = True)
    


    

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
    x_pos = np.arange(len(regions))
    regions2 = np.array([A500,B500])

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


















def checkCBSWithin(cold, fullClass, reducedClass, noObs):
    
    
    #Get the data for the cold similarity ratings
    print('cold sim descriptives:')
    dfCold = pd.read_csv(cold)
    dfCold = dfCold.replace(' Beta', 'Beta')

    dfColdA = dfCold.drop(index = dfCold[dfCold['category1'] != 'Alpha'].index)
    dfColdA = dfColdA.drop(index = dfColdA[dfColdA['category2'] != 'Alpha'].index)
    print(dfColdA['response'].mean(), dfColdA['response'].std()) #average within category similarity

    dfColdB = dfCold.drop(index = dfCold[dfCold['category1'] != 'Beta'].index)
    dfColdB = dfColdB.drop(index = dfColdB[dfColdB['category2'] != 'Beta'].index)
    print(dfColdB['response'].mean(), dfColdB['response'].std()) #average within category similarity

    dfColdG = dfCold.drop(index = dfCold[dfCold['category1'] != 'Gamma'].index)
    dfColdG = dfColdG.drop(index = dfColdG[dfColdG['category2'] != 'Gamma'].index)
    print(dfColdG['response'].mean(), dfColdG['response'].std()) #average within category similarity
    print('                  ')





    #get the data for the full classification condition similarity ratings
    print('full calss sim descriptives:')
    dfFull = pd.read_csv(fullClass)

    dfFull = dfFull.replace(' Beta', 'Beta')
  

    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1450'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1450'].index)

    dfFullA = dfFull.drop(index = dfFull[dfFull['category1'] != 'Alpha'].index)
    dfFullA = dfFullA.drop(index = dfFullA[dfFullA['category2'] != 'Alpha'].index)
    print(dfFullA['response'].mean(), dfFullA['response'].std()) #average within category similarity

    dfFullB = dfFull.drop(index = dfFull[dfFull['category1'] != 'Beta'].index)
    dfFullB = dfFullB.drop(index = dfFullB[dfFullB['category2'] != 'Beta'].index)
    print(dfFullB['response'].mean(), dfFullB['response'].std()) #average within category similarity

    dfFullG = dfFull.drop(index = dfFull[dfFull['category1'] != 'Gamma'].index)
    dfFullG = dfFullG.drop(index = dfFullG[dfFullG['category2'] != 'Gamma'].index)
    print(dfFullG['response'].mean(), dfFullG['response'].std()) #average within category similarity
    print('                  ')






    #get the data for the reduced classification condition similarity ratings
    print('reduced calss sim descriptives:')
    dfReduced = pd.read_csv(reducedClass)

    dfReduced = dfReduced.replace(' Beta', 'Beta')




    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1350'].index)
    dfREduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1450'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1450'].index)

    dfReducedA = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Alpha'].index)
    dfReducedA = dfReducedA.drop(index = dfReducedA[dfReducedA['category2'] != 'Alpha'].index)
    print(dfReducedA['response'].mean(), dfReducedA['response'].std()) #average within category similarity

    dfReducedB = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Beta'].index)
    dfReducedB = dfReducedB.drop(index = dfReducedB[dfReducedB['category2'] != 'Beta'].index)
    print(dfReducedB['response'].mean(), dfReducedB['response'].std()) #average within category similarity

    dfReducedG = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Gamma'].index)
    dfReducedG = dfReducedG.drop(index = dfReducedG[dfReducedG['category2'] != 'Gamma'].index)
    print(dfReducedG['response'].mean(), dfReducedG['response'].std()) #average within category similarity
    print('                  ')







    #get the data for the no observation condition similarity data
    print('no obs calss sim descriptives:')
    dfNoObs = pd.read_csv(noObs)

    dfNoObs = dfNoObs.replace(' Beta', 'Beta')


    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1450'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1450'].index)


    dfNoObsA = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Alpha'].index)
    dfNoObsA = dfNoObsA.drop(index = dfNoObsA[dfNoObsA['category2'] != 'Alpha'].index)
    print(dfNoObsA['response'].mean(), dfNoObsA['response'].std()) #average within category similarity

    dfNoObsB = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Beta'].index)
    dfNoObsB = dfNoObsB.drop(index = dfNoObsB[dfNoObsB['category2'] != 'Beta'].index)
    print(dfNoObsB['response'].mean(), dfNoObsB['response'].std()) #average within category similarity

    dfNoObsG = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Gamma'].index)
    dfNoObsG = dfNoObsG.drop(index = dfNoObsG[dfNoObsG['category2'] != 'Gamma'].index)
    print(dfNoObsG['response'].mean(), dfNoObsG['response'].std()) #average within category similarity
    print('                  ')


    #Graph the ALPHAS
    regions = np.array(['Cold Similarity', 'Full Class', 'Reduced Class', 'No Observation'])
    print(regions,"regions")
    x_pos = np.arange(len(regions))

    x_pos2 = np.arange(len(regions))

    # Build the plot
    fig1, ax1 = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax1.bar(x_pos[0], dfColdA['response'].mean(),yerr=sem(dfColdA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='.')
    ax1.bar(x_pos[1], dfFullA['response'].mean(),yerr=sem(dfFullA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='.')
    ax1.bar(x_pos[2], dfReducedA['response'].mean(),yerr=sem(dfReducedA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='.')
    ax1.bar(x_pos[3], dfNoObsA['response'].mean(),yerr=sem(dfNoObsA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='.')
    ax1.set_ylim(0,6)
    ax1.set_ylabel('Average within Alpha category similarity')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regions)
    ax1.set_title("Alpha similarity ratings across conditions")
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)

    fig1.savefig('Analysis/graphs/AlphaSimilarity.png')
    #plt.show()

    #Graph the BETAS
    fig2, ax2 = plt.subplots()

    ax2.bar(x_pos[0], dfColdB['response'].mean(),yerr=sem(dfColdB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[1], dfFullB['response'].mean(),yerr=sem(dfFullB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[2], dfReducedB['response'].mean(),yerr=sem(dfReducedB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[3], dfNoObsB['response'].mean(),yerr=sem(dfNoObsB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.set_ylim(0,6)
    
    ax2.set_ylabel('Average within Beta category similarity')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regions)
    ax2.set_title("Beta similarity ratings across conditions")
    ax2.yaxis.grid(False)
    ax2.tick_params(axis='x', which='major', labelsize=10)
    fig2.savefig('Analysis/graphs/BetaSimilarity.png')



    #Graph the GAMMAS
    fig3, ax3 = plt.subplots()


    ax3.bar(x_pos[0], dfColdG['response'].mean(),yerr=sem(dfColdG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[1], dfFullG['response'].mean(),yerr=sem(dfFullG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[2], dfReducedG['response'].mean(),yerr=sem(dfReducedG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[3], dfNoObsG['response'].mean(),yerr=sem(dfNoObsG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.set_ylim(0,6)
    
    ax3.set_ylabel('Average within Gamma category similarity')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(regions)
    ax3.set_title("Gamma similarity ratings across conditions")
    ax3.yaxis.grid(False)
    ax3.tick_params(axis='x', which='major', labelsize=10)
    fig3.savefig('Analysis/graphs/GammaSimilarity.png')

    #quick anaova with scipy because more simple to set up than the stats models ols
    print('ANOVAs')
    print(f_oneway(dfColdA['response'], dfFullA['response'], dfReducedA['response'], dfNoObsA['response']))
    print('                         ')
    print(f_oneway(dfColdB['response'], dfFullB['response'], dfReducedB['response'], dfNoObsB['response']))
    print('                         ')
    print(f_oneway(dfColdG['response'], dfFullG['response'], dfReducedG['response'], dfNoObsG['response']))
    print('                         ')

    #now that I know the ALPHA anova is sig, anova with statsmodels ols so I can get partial eta squared
    olsdata = np.array(list(dfColdA['response']) + list(dfFullA['response']) + list(dfReducedA['response']) + list(dfNoObsA['response']))
    olslabels = (['ColdA'] * len(dfColdA['response'])) + (['FullA'] * len(dfFullA['response']))+ (['ReducedA'] * len(dfReducedA['response'])) + (['NoObsA'] * len(dfNoObsA['response']))
    olsdf = pd.DataFrame({'data': olsdata, 'group': olslabels})
    model = ols('data ~ C(group)', data=olsdf).fit()
    anova_table = sm.stats.anova_lm(model)
    print(anova_table)
    # Calculate eta-squared
    ss_between = anova_table['sum_sq'][0]
    ss_total = ss_between + anova_table['sum_sq'][1]
    eta_squared = ss_between / ss_total

    print("Eta-squared value:", eta_squared)


    print('POST HOC THSD on ALPHAS')
    posthoc = tukey_hsd(dfColdA['response'], dfFullA['response'], dfReducedA['response'], dfNoObsA['response'])
    print(posthoc)
    effectSize = cohend(d1=dfColdA['response'], d2=dfReducedA['response'])
    print(effectSize)
    
    print(len(dfCold['id'].unique()))













def checkCBSBetween(cold, fullClass, reducedClass, noObs):
    
    
    #Get the data for the cold similarity ratings
    print('cold sim descriptives:')
    dfCold = pd.read_csv(cold)

    dfCold = dfCold.replace(' Beta', 'Beta')

    dfColdA = dfCold.drop(index = dfCold[dfCold['category1'] != 'Alpha'].index)
    dfColdA = dfColdA.drop(index = dfColdA[dfColdA['category2'] == 'Alpha'].index)
    print(dfColdA['response'].mean(), dfColdA['response'].std()) #average Between category similarity

    dfColdB = dfCold.drop(index = dfCold[dfCold['category1'] != 'Beta'].index)
    dfColdB = dfColdB.drop(index = dfColdB[dfColdB['category2'] == 'Beta'].index)
    print(dfColdB['response'].mean(), dfColdB['response'].std()) #average between category similarity

    dfColdG = dfCold.drop(index = dfCold[dfCold['category1'] != 'Gamma'].index)
    dfColdG = dfColdG.drop(index = dfColdG[dfColdG['category2'] == 'Gamma'].index)
    print(dfColdG['response'].mean(), dfColdG['response'].std()) #average between category similarity
    print('                  ')





    #get the data for the full classification condition similarity ratings
    print('full calss sim descriptives:')
    dfFull = pd.read_csv(fullClass)

    dfFull = dfFull.replace(' Beta', 'Beta')

    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1450'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1450'].index)

    dfFullA = dfFull.drop(index = dfFull[dfFull['category1'] != 'Alpha'].index)
    dfFullA = dfFullA.drop(index = dfFullA[dfFullA['category2'] == 'Alpha'].index)
    print(dfFullA['response'].mean(), dfFullA['response'].std()) #average between category similarity

    dfFullB = dfFull.drop(index = dfFull[dfFull['category1'] != 'Beta'].index)
    dfFullB = dfFullB.drop(index = dfFullB[dfFullB['category2'] == 'Beta'].index)
    print(dfFullB['response'].mean(), dfFullB['response'].std()) #average between category similarity

    dfFullG = dfFull.drop(index = dfFull[dfFull['category1'] != 'Gamma'].index)
    dfFullG = dfFullG.drop(index = dfFullG[dfFullG['category2'] == 'Gamma'].index)
    print(dfFullG['response'].mean(), dfFullG['response'].std()) #average between category similarity
    print('                  ')






    #get the data for the reduced classification condition similarity ratings
    print('reduced calss sim descriptives:')
    dfReduced = pd.read_csv(reducedClass)
    dfReduced = dfReduced.replace(' Beta', 'Beta')


    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1350'].index)
    dfREduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1450'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1450'].index)

    dfReducedA = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Alpha'].index)
    dfReducedA = dfReducedA.drop(index = dfReducedA[dfReducedA['category2'] == 'Alpha'].index)
    print(dfReducedA['response'].mean(), dfReducedA['response'].std()) #average between category similarity

    dfReducedB = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Beta'].index)
    dfReducedB = dfReducedB.drop(index = dfReducedB[dfReducedB['category2'] == 'Beta'].index)
    print(dfReducedB['response'].mean(), dfReducedB['response'].std()) #average between category similarity

    dfReducedG = dfReduced.drop(index = dfReduced[dfReduced['category1'] != 'Gamma'].index)
    dfReducedG = dfReducedG.drop(index = dfReducedG[dfReducedG['category2'] == 'Gamma'].index)
    print(dfReducedG['response'].mean(), dfReducedG['response'].std()) #average between category similarity
    print('                  ')







    #get the data for the no observation condition similarity data
    print('no obs calss sim descriptives:')
    dfNoObs = pd.read_csv(noObs) 
    dfNoObs = dfNoObs.replace(' Beta', 'Beta')

    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1450'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1450'].index)


    dfNoObsA = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Alpha'].index)
    dfNoObsA = dfNoObsA.drop(index = dfNoObsA[dfNoObsA['category2'] == 'Alpha'].index)
    print(dfNoObsA['response'].mean(), dfNoObsA['response'].std()) #average between category similarity

    dfNoObsB = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Beta'].index)
    dfNoObsB = dfNoObsB.drop(index = dfNoObsB[dfNoObsB['category2'] == 'Beta'].index)
    print(dfNoObsB['response'].mean(), dfNoObsB['response'].std()) #average between category similarity

    dfNoObsG = dfNoObs.drop(index = dfNoObs[dfNoObs['category1'] != 'Gamma'].index)
    dfNoObsG = dfNoObsG.drop(index = dfNoObsG[dfNoObsG['category2'] == 'Gamma'].index)
    print(dfNoObsG['response'].mean(), dfNoObsG['response'].std()) #average between category similarity
    print('                  ')


    #Graph the ALPHAS
    regions = np.array(['Cold Similarity', 'Full Class', 'Reduced Class', 'No Observation'])
    print(regions,"regions")
    x_pos = np.arange(len(regions))

    x_pos2 = np.arange(len(regions))

    # Build the plot
    fig1, ax1 = plt.subplots()
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax1.bar(x_pos[0], dfColdA['response'].mean(),yerr=sem(dfColdA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch= '.')
    ax1.bar(x_pos[1], dfFullA['response'].mean(),yerr=sem(dfFullA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch= '.')
    ax1.bar(x_pos[2], dfReducedA['response'].mean(),yerr=sem(dfReducedA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch= '.')
    ax1.bar(x_pos[3], dfNoObsA['response'].mean(),yerr=sem(dfNoObsA['response']), align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch= '.')
    ax1.set_ylim(0,6)
    ax1.set_ylabel('Average Between Alpha category similarity')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regions)
    ax1.set_title("Alpha: Between Category Similarity Across Conditions")
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)

    fig1.savefig('Analysis/graphs/BetweenAlphaSimilarity.png')
    #plt.show()

    #Graph the BETAS
    fig2, ax2 = plt.subplots()

    ax2.bar(x_pos[0], dfColdB['response'].mean(),yerr=sem(dfColdB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[1], dfFullB['response'].mean(),yerr=sem(dfFullB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[2], dfReducedB['response'].mean(),yerr=sem(dfReducedB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.bar(x_pos[3], dfNoObsB['response'].mean(),yerr=sem(dfNoObsB['response']), align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='/')
    ax2.set_ylim(0,6)
    
    ax2.set_ylabel('Average Between Beta category similarity')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regions)
    ax2.set_title("Beta: Between Category Similarity Across Conditions")
    ax2.yaxis.grid(False)
    ax2.tick_params(axis='x', which='major', labelsize=10)
    fig2.savefig('Analysis/graphs/BetweenBetaSimilarity.png')



    #Graph the GAMMAS
    fig3, ax3 = plt.subplots()


    ax3.bar(x_pos[0], dfColdG['response'].mean(),yerr=sem(dfColdG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[1], dfFullG['response'].mean(),yerr=sem(dfFullG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[2], dfReducedG['response'].mean(),yerr=sem(dfReducedG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.bar(x_pos[3], dfNoObsG['response'].mean(),yerr=sem(dfNoObsG['response']), align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='k', capsize=4, width=0.7, hatch='o')
    ax3.set_ylim(0,6)
    
    ax3.set_ylabel('Average Between Gamma category similarity')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(regions)
    ax3.set_title("Gamma: Between Category Similarity Across Conditions")
    ax3.yaxis.grid(False)
    ax3.tick_params(axis='x', which='major', labelsize=10)
    fig3.savefig('Analysis/graphs/BetweenGammaSimilarity.png')

    #do scipy anovas fisrt because they're simple to set up
    print('ANOVAs')
    print(f_oneway(dfColdA['response'], dfFullA['response'], dfReducedA['response'], dfNoObsA['response']))
    print('                         ')
    print(f_oneway(dfColdB['response'], dfFullB['response'], dfReducedB['response'], dfNoObsB['response']))
    print('                         ')
    print(f_oneway(dfColdG['response'], dfFullG['response'], dfReducedG['response'], dfNoObsG['response']))
    print('                         ')


    #now that I know the BETA anova is sig, anova with statsmodels ols so I can get partial eta squared
    olsdata = np.array(list(dfColdB['response']) + list(dfFullB['response']) + list(dfReducedB['response']) + list(dfNoObsB['response']))
    olslabels = (['ColdB'] * len(dfColdB['response'])) + (['FullB'] * len(dfFullB['response']))+ (['ReducedB'] * len(dfReducedB['response'])) + (['NoObsB'] * len(dfNoObsB['response']))
    olsdf = pd.DataFrame({'data': olsdata, 'group': olslabels})
    model = ols('data ~ C(group)', data=olsdf).fit()
    anova_table = sm.stats.anova_lm(model)
    print(anova_table)
    # Calculate eta-squared
    ss_between = anova_table['sum_sq'][0]
    ss_total = ss_between + anova_table['sum_sq'][1]
    eta_squared = ss_between / ss_total

    print("Eta-squared value:", eta_squared)


    print('POST HOC on BETAS')
    M = np.array([dfColdB['response'].mean(),dfFullB['response'].mean(), dfReducedB['response'].mean(), dfNoObsB['response'].mean()])
    var = np.array([dfColdB['response'].sem(),dfFullB['response'].sem(), dfReducedB['response'].sem(), dfNoObsB['response'].sem()])

    comparisons= [dfColdB['response'], dfFullB['response'], dfReducedB['response'], dfNoObsB['response']]
    
    posthoc = skpost.posthoc_scheffe(a=comparisons)
    print(posthoc)
    effectSize = cohend(d1=dfFullB['response'], d2=dfReducedB['response'])
    print(effectSize)
    









'''
cold= 'cold_sim_results.csv'
fullClass= 'cond_2_similarity_results.csv'
reducedClass = 'cond_1_similarity_results.csv'
noObs= 'cond_3_similarity_results.csv'
checkCBSBetween(
    cold=cold,
    fullClass=fullClass,
    reducedClass=reducedClass,
    noObs=noObs
    )'''





def checkThatAccisCorrect(path):
    df = pd.read_csv(path)


    dfA= df.drop(index = df[df['response'] != 'Alpha'].index)
    dfA= dfA.drop(index = dfA[dfA['category'] != 'Alpha'].index)


    dfB= df.drop(index = df[df['response'] != 'Beta'].index)
    #dfB= dfB.drop(index = dfB[dfB['category'] != 'Beta'].index)
    dfB= dfB.drop(index = dfB[dfB['stimId'] != 'B950'].index)


    #print(dfA['accuracy'])
    print(dfB)


#checkThatAccisCorrect(path='cond_3_testPhase_results.csv')



















##find the 4 different supprofiles for typicality<-- (1) positive slope, (2) negative slope, (3) flat gradient, (4) central tendency


def subTypeTyps(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])

    numSubs= len(df['id'].unique())


    central = np.array([2,4,6,4,2])
    pos = np.array([2,3,4,5,6])
    neg = np.array([6,5,4,3,2])
    flat = np.array([5,5,5,5,5])
    regions = np.array(['1', '2', '3', '4', '5']) 
    x_pos = np.arange(len(regions))

    '''
    # Build the plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharex=True, sharey=True)
    #plt.figure(figsize=(20, 3))
    #plt.ylim(0,1)
    ax1.bar(x_pos, central, align='center',color='gray',edgecolor='black',linewidth=3, alpha=0.6, ecolor='black', capsize=4, width=0.7, )
    ax2.bar(x_pos, pos, align='center',color='gray',edgecolor='black',linewidth=3, alpha=0.6, ecolor='black', capsize=4, width=0.7,)
    ax3.bar(x_pos, neg, align='center',color='gray',edgecolor='black',linewidth=3, alpha=0.6, ecolor='black', capsize=4, width=0.7,)
    ax4.bar(x_pos, flat, align='center',color='gray',edgecolor='black',linewidth=3, alpha=0.6, ecolor='black', capsize=4, width=0.7,)
    
    ax1.set_ylim(0,7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regions)
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='major', labelsize=10)
    ax2.set_ylim(0,7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regions)
    ax2.yaxis.grid(False)
    ax2.tick_params(axis='x', which='major', labelsize=10)
    ax3.set_ylim(0,7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(regions)
    ax3.yaxis.grid(False)
    ax3.tick_params(axis='x', which='major', labelsize=10)
    ax4.set_ylim(0,7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(regions)
    ax4.yaxis.grid(False)
    ax4.tick_params(axis='x', which='major', labelsize=10)


    ax1.set_title('Central Tendency',fontsize=8)
    ax2.set_title('Positive Slope', fontsize=8)
    ax3.set_title('Negative Slope', fontsize = 8)
    ax4.set_title('Flat Gradient', fontsize = 8)
    fig.suptitle('Possible Subprofiles', fontsize = 20)
    fig.supylabel('Typicality')
    fig.supxlabel('Category Exemplars')

    plt.savefig('Analysis/graphs/individual_typs/PossibleSubTypes.png')
    plt.clf()'''


    
    for subject in df['id'].unique():
        means_df = df[df['id']== subject].groupby(['stimId'], as_index= True)['response'].mean()

        print(subject)
        print(np.array(means_df.index))
        means = np.array(means_df)
        regions = np.array(means_df.index)
        x_pos = np.arange(len(regions))
    
        # Build the plot
        fig, ax = plt.subplots()
        #plt.figure(figsize=(20, 3))
        #plt.ylim(0,1)
        ax.bar(x_pos[0:5], means[0:5], align='center',color='r',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '.')
        ax.bar(x_pos[5:10], means[5:10], align='center',color='b',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= '/')
        ax.bar(x_pos[10:15], means[10:15], align='center',color='g',edgecolor='black',linewidth=3, alpha=0.9, ecolor='black', capsize=4, width=0.7, hatch= 'o')
        ax.set_ylim(0,7)
        ax.set_ylabel('Typicality')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions)
        ax.set_title(str(title)+" subject "+str(subject))
        ax.yaxis.grid(False)
        ax.tick_params(axis='x', which='major', labelsize=10)

        plt.savefig('Analysis/graphs/individual_typs/'+'ss'+str(subject)+'_'+str(title) + '.png')
        











changes= {
    'A50': 'A1',
    'A150': 'A2',
    'A250': 'A3',
    'A350': 'A4',
    'A450': 'A5',

    'B550': 'B1',
    'B650': 'B2',
    'B750': 'B3',
    'B850': 'B4',
    'B950': 'B5',

    'C1050': 'G1',
    'C1150': 'G2',
    'C1250': 'G3',
    'C1350': 'G4',
    'C1450': 'G5',
}

subTypeTyps(path='C:/Users/apers/line_data/cond_3_typicality2_results.csv', title= 'No Observation Typicality', changestimID=True,changes=changes)


