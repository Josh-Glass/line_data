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


def AggLearnDescriptives(path, title):
    df = pd.read_csv(path)
    


    df['accuracy'] = df['accuracy'].astype(str)
    df = df.replace('True', 1)
    df = df.replace('False', 0)


    #GET OVERALL LEARN CURVES
    means_df = df.groupby(['block'], as_index= True)['accuracy'].describe()
    means_df.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    means_df.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_AggData.csv')







def PerCatLearnDescriptives(path, title, changestimID, changes):
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




    #GET PERCAT LEARN CURVES (ALL THREE CATS ON ONE GRAPH)
    #alpha cat data
    means_dfA = dfA.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfA.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    means_dfA.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataCatA.csv')


    #beta cat data
    means_dfB = dfB.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfB.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    means_dfB.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataCatB.csv')


    #gamma cat
    means_dfG = dfG.groupby(['block'], as_index= True)['accuracy'].describe()
    means_dfG.rename(columns = {'mean':'means', 'std':'stndDev'}, inplace = True)
    means_dfG.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataCatG.csv')


    #GET PER ITEM LEARN CUREVES (EACH CAT ON SEPARATE PLOT)

    item_means_dfA = dfA.groupby(['stimId','block'], as_index= True)['accuracy'].describe()
    item_means_dfA.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataAs.csv')


    item_means_dfB = dfB.groupby(['stimId','block'], as_index= True)['accuracy'].describe()
    item_means_dfB.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataBs.csv')

    item_means_dfG = dfG.groupby(['stimId','block'], as_index= True)['accuracy'].describe()
    item_means_dfG.to_csv('Analysis/stats_output/'+str(title)+'_Learning_Curve_DataGs.csv')

     


        














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
    print('Overall: ', df['accuracy'].mean(), df['accuracy'].std())
    print('Alpha: ',df[df['category']=='Alpha']['accuracy'].mean(), ',', df[df['category']=='Alpha']['accuracy'].std())
    print('Beta: ',df[df['category']=='Beta']['accuracy'].mean(), ',', df[df['category']=='Beta']['accuracy'].std())
    print('Gamma: ',df[df['category']=='Gamma']['accuracy'].mean(), ',', df[df['category']=='Gamma']['accuracy'].std())
    print(f_oneway(df[df['category']=='Alpha']['accuracy'],df[df['category']=='Beta']['accuracy'],df[df['category']=='Gamma']['accuracy']))
    print(skpost.posthoc_scheffe(a=[df[df['category']=='Alpha']['accuracy'],df[df['category']=='Beta']['accuracy'],df[df['category']=='Gamma']['accuracy']]))
    print('              ')





    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])



    

    means = np.array(df.groupby(['stimId'], as_index= True)['accuracy'].mean())
    #print(means)
    sem = np.array(df.groupby(['stimId'], as_index= True)['accuracy'].sem())

    
    







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




















def HardTestRight_GetGens(path, title):
    df = pd.read_csv(str(path))
    df = df.replace(' Beta', 'Beta')

    for index, row in df.iterrows():
        if row['response'] == row['category']:
            df.at[index, 'accuracy'] = 1

    dfFailedAPnums = df[df['stimId']=='A450']
    dfFailedAPnums = dfFailedAPnums.drop(index = dfFailedAPnums[dfFailedAPnums['response'] == 'Alpha'].index)
    failedAPnums= list(dfFailedAPnums['id'])
    #print(dfFailedAPnums)

    dfA = df
    for item in failedAPnums:
        dfA = dfA.drop(index = dfA[dfA['id'] == item].index)


    dfFailedBPnums = df[df['stimId']=='B950']
    #print(dfFailedBPnums['accuracy'].mean())
    dfFailedBPnums = dfFailedBPnums.drop(index = dfFailedBPnums[dfFailedBPnums['response'] == 'Beta'].index)
    failedBPnums= list(dfFailedBPnums['id'])
    #print(dfFailedBPnums)

    dfB = df
    for item in failedBPnums:
        dfB = dfB.drop(index = dfB[dfB['id'] == item].index)


    newdfB = dfB.drop(index = dfB[dfB['category'] != 'test'].index)
    groupB= newdfB.groupby(['stimId','response'], as_index= True)['id'].describe()
    groupB.rename(columns = {'count':'counts'}, inplace = True)
    print(groupB)


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
    












