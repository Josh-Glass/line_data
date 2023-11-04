import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from scipy.stats import ttest_rel
import scipy.stats as stats

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

     


        














def TestAccDescriptives(path, title, changestimID, changes):
    df = pd.read_csv(path)
    #fix the weird Beta thing
    df = df.replace(' Beta', 'Beta')
    for index, row in df.iterrows():
        if row['response'] == row['category']:
            df.at[index, 'accuracy'] = 1

    #make sure acc is an int
    df = df.replace('True', 1)
    df = df.replace('False', 0)
    df = df.drop(index = df[df['category'] == 'test'].index)

    

    meansdf = df.groupby(['category'], as_index= True)['accuracy'].describe()
    meansdf.to_csv('Analysis/stats_output/'+str(title)+'_CatTestAcc.csv')


    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])


    item_means = df.groupby(['stimId'], as_index= True)['accuracy'].describe()
    meansdf.to_csv('Analysis/stats_output/'+str(title)+'_ItemTestAcc.csv')


    
    






def TypsDescriptives(path, title, changestimID, changes):
    df = pd.read_csv(path)
    if changestimID == True:
        for i in changes:
            df= df.replace(i, changes[i])
    means_df = df.groupby(['stimId'], as_index= True)['response'].describe()
    means_df.to_csv('Analysis/stats_output/'+str(title)+'_Descriptives.csv')







def prePostTypReg(prePath,postPath, title, changestimID, changes):
    predf = pd.read_csv(prePath)
    postdf = pd.read_csv(postPath)



    if changestimID == True:
        for i in changes:
            predf= predf.replace(i, int(changes[i]))
            postdf= postdf.replace(i, int(changes[i]))

    postdfA = postdf.drop(index = postdf[postdf['category'] != 'Alpha'].index)
    

    
    preSlopes=[]
    for subject in predf['id'].unique():
        Y= np.array(predf[predf['id']==subject]['response'])        
        X =np.array(list(predf[predf['id']==subject]['stimId']))
        X = np.log10(X)
        X = sm.add_constant(X)
       

        model= sm.OLS(Y,X)
        results = model.fit()
        preSlopes.append(results.params[1])
                






    postSlopes=[]
    for subject in postdfA['id'].unique():
        Y= np.array(postdfA[postdfA['id']==subject]['response'])        
        X =np.array(list(postdfA[postdfA['id']==subject]['stimId']))
        X = np.log10(X)
        X = sm.add_constant(X)

        model= sm.OLS(Y,X)
        results = model.fit()
        postSlopes.append(results.params[1])


    ttestRes = ttest_rel(preSlopes,postSlopes, #alternative='greater'
                         )
    print(ttestRes)

    resultsdf = pd.DataFrame({
        'preSlopes Mean': np.array(preSlopes).mean(),
        'preSlopes std':  np.array(preSlopes).std(),
        'postSlopes Mean': np.array(postSlopes).mean(),
        'postSlopes std':  np.array(postSlopes).std(),
        'ttest stat' : ttestRes[0],
        'pval' : ttestRes[1],
        'numsubs': len(list(predf['id'].unique()))}, index=[0])
    
    print(resultsdf)

    resultsdf.to_csv('Analysis/stats_output/'+str(title)+'_reg.csv')








changes= {
    'A50': '50',
    'A150': '150',
    'A250': '250',
    'A350': '350',
    'A450': '450',
}
#prePostTypReg(prePath='cond_2_typicality1_results.csv', postPath='cond_2_typicality2_results.csv', title='FullClass', changestimID=True, changes=changes)






def GetGens(path, title):
    df = pd.read_csv(str(path))
    newdf = df.drop(index = df[df['category'] != 'test'].index)
    group= newdf.groupby(['stimId','response'], as_index= True)['id'].describe()
    group.rename(columns = {'count':'counts'}, inplace = True)
    group.to_csv('Analysis/stats_output/'+'Overall_'+str(title)+'.csv')



















def HardTestStats(pathtest,pathlearn, title):
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

    dfA = df
    for item in failedAPnums:
        dfA = dfA.drop(index = dfA[dfA['id'] == item].index)


    dfFailedBPnums = df[df['stimId']=='B950']
    dfFailedBPnums = dfFailedBPnums.drop(index = dfFailedBPnums[dfFailedBPnums['response'] == 'Beta'].index)
    failedBPnums= list(dfFailedBPnums['id']) + learnFailB

    dfB = df
    for item in failedBPnums:
        dfB = dfB.drop(index = dfB[dfB['id'] == item].index)


    newdfB = dfB.drop(index = dfB[dfB['category'] != 'test'].index)
    groupB= newdfB.groupby(['stimId','response'], as_index= True)['id'].describe()
    groupB.rename(columns = {'count':'counts'}, inplace = True)
    groupB.to_csv('Analysis/stats_output/'+'HardB_'+str(title)+'.csv')

    newdfA = dfA.drop(index = dfA[dfA['category'] != 'test'].index)
    groupA= newdfA.groupby(['stimId','response'], as_index= True)['id'].describe()
    groupA.rename(columns = {'count':'counts'}, inplace = True)
    groupA.to_csv('Analysis/stats_output/'+'HardA_'+str(title)+'.csv')



    countsA= np.array(groupA['counts'])
    numsubsA = len(np.array(dfA['id'].unique()))
    dataA = countsA
    AT500=dataA[2]
    BT500=dataA[3]



    countsB= np.array(groupB['counts'])
    numsubsB = len(np.array(dfB['id'].unique()))
    dataB = countsB
    BT1000 = dataB[0]
    GT1000 = dataB[1]


    table1 = np.array([[AT500, (numsubsA-AT500)], [BT500, (numsubsA-BT500)]])
    res500=stats.fisher_exact(table1, alternative='less')


    table2 = np.array([[BT1000, (numsubsB-BT1000)], [BT1000, (numsubsB-BT1000)]])
    res1000= stats.fisher_exact(table2, alternative='less')



    resdf = pd.DataFrame([{
        'T500 pval' : res500[1],
        'T1000 pval' : res1000[1]
    }], index=[0])

    resdf.to_csv('Analysis/stats_output/'+'FisherExact'+str(title)+'.csv')
    print(resdf)
   




#HardTestStats(pathlearn='cond_1_trainPhase_results.csv', pathtest='cond_1_testPhase_results.csv',title='at Train And Test (Reduced Classification)')



   










def checkCBSWithin_BoundaryItems(cold, fullClass, reducedClass, noObs):
    
    
    #Get the data for the cold similarity ratings
    print('cold sim descriptives:')
    dfCold = pd.read_csv(cold)
    dfCold = dfCold.replace(' Beta', 'Beta')

    dfColdA = dfCold.drop(index = dfCold[dfCold['stimId1'] != 'A50'].index)
    dfColdA = dfColdA.drop(index = dfColdA[dfColdA['stimID2'] != 'A450'].index)
    print(dfColdA['response'].mean())

    dfColdB = dfCold.drop(index = dfCold[dfCold['stimId1'] != 'B550'].index)
    dfColdB = dfColdB.drop(index = dfColdB[dfColdB['stimID2'] != 'B950'].index)
    print(dfColdB['response'].mean())

    dfColdG = dfCold.drop(index = dfCold[dfCold['stimId1'] != 'C1050'].index)
    dfColdG = dfColdG.drop(index = dfColdG[dfColdG['stimID2'] != 'C1250'].index)
    print(dfColdG['response'].mean())

    
    coldMeans= [dfColdA['response'].mean(),dfColdB['response'].mean(), dfColdG['response'].mean()]
    coldSD = [dfColdA['response'].std(),dfColdB['response'].std(), dfColdG['response'].std()]



    #get the data for the full classification condition similarity ratings
    print('full calss sim descriptives:')
    dfFull = pd.read_csv(fullClass)

    dfFull = dfFull.replace(' Beta', 'Beta')
  

    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1450'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1450'].index)

    dfFullA = dfFull.drop(index = dfFull[dfFull['stimId1'] != 'A50'].index)
    dfFullA = dfFullA.drop(index = dfFullA[dfFullA['stimID2'] != 'A450'].index)
    print(dfFullA['response'].mean())



    dfFullB = dfFull.drop(index = dfFull[dfFull['stimId1'] != 'B550'].index)
    dfFullB = dfFullB.drop(index = dfFullB[dfFullB['stimID2'] != 'B950'].index)
    print(dfFullB['response'].mean())

    dfFullG = dfFull.drop(index = dfFull[dfFull['stimId1'] != 'C1050'].index)
    dfFullG = dfFullG.drop(index = dfFullG[dfFullG['stimID2'] != 'C1250'].index)
    print(dfFullG['response'].mean())

    FullMeans= [dfFullA['response'].mean(),dfFullB['response'].mean(), dfFullG['response'].mean()]
    FullSD = [dfFullA['response'].std(),dfFullB['response'].std(), dfFullG['response'].std()]







    #get the data for the reduced classification condition similarity ratings
    print('reduced calss sim descriptives:')
    dfReduced = pd.read_csv(reducedClass)

    dfReduced = dfReduced.replace(' Beta', 'Beta')




    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1450'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1450'].index)

    dfReducedA = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] != 'A50'].index)
    dfReducedA = dfReducedA.drop(index = dfReducedA[dfReducedA['stimID2'] != 'A450'].index)
    print(dfReducedA['response'].mean())

    dfReducedB = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] != 'B550'].index)
    dfReducedB = dfReducedB.drop(index = dfReducedB[dfReducedB['stimID2'] != 'B950'].index)
    print(dfReducedB['response'].mean())

    dfReducedG = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] != 'C1050'].index)
    dfReducedG = dfReducedG.drop(index = dfReducedG[dfReducedG['stimID2'] != 'C1250'].index)
    print(dfReducedG['response'].mean())


    ReducedMeans= [dfReducedA['response'].mean(),dfReducedB['response'].mean(), dfReducedG['response'].mean()]
    ReducedSD = [dfReducedA['response'].std(),dfReducedB['response'].std(), dfReducedG['response'].std()]





    #get the data for the no observation condition similarity data
    print('no obs calss sim descriptives:')
    dfNoObs = pd.read_csv(noObs)

    dfNoObs = dfNoObs.replace(' Beta', 'Beta')


    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1450'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1450'].index)


    dfNoObsA = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] != 'A50'].index)
    dfNoObsA = dfNoObsA.drop(index = dfNoObsA[dfNoObsA['stimID2'] != 'A450'].index)
    print(dfNoObsA['response'].mean())

    dfNoObsB = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] != 'B550'].index)
    dfNoObsB = dfNoObsB.drop(index = dfNoObsB[dfNoObsB['stimID2'] != 'B950'].index)
    print(dfNoObsB['response'].mean())

    dfNoObsG = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] != 'C1050'].index)
    dfNoObsG = dfNoObsG.drop(index = dfNoObsG[dfNoObsG['stimID2'] != 'C1250'].index)
    print(dfNoObsG['response'].mean())



    NoObsMeans= [dfNoObsA['response'].mean(), dfNoObsB['response'].mean(), dfNoObsG['response'].mean()]
    NoObsSD = [dfNoObsA['response'].std(), dfNoObsB['response'].std(), dfNoObsG['response'].std()]






    numsubs = len(dfCold['id'].unique()) + len(dfFull['id'].unique()) +len(dfReduced['id'].unique()) + len(dfNoObs['id'].unique())

    #quick anaova with scipy because more simple to set up than the stats models ols
    print('ANOVAs')
    Astat, Apval= f_oneway(dfColdA['response'], dfFullA['response'], dfReducedA['response'], dfNoObsA['response'])

    Bstat, Bpval= f_oneway(dfColdB['response'], dfFullB['response'], dfReducedB['response'], dfNoObsB['response'])
    
    
    Gstat, Gpval= f_oneway(dfColdG['response'], dfFullG['response'], dfReducedG['response'], dfNoObsG['response'])

    output = pd.DataFrame(
        {
            'Category' : ['Alpha', 'Beta', 'Gamma'],
            'Cold Means': coldMeans,
            'Cold SD' : coldSD,
            'Full Means' : FullMeans,
            'Full SD' : FullSD,
            'Reduced Means' : ReducedMeans,
            'Reduced SD' : ReducedSD,
            'No Obs Means' : NoObsMeans,
            'No Obs SD' : NoObsSD,
            'F statistic' : [Astat, Bstat, Gstat],
            'p value' : [Apval, Bpval, Gpval],
            'n': [numsubs,numsubs,numsubs]
        },
    )

    output.to_csv('Analysis/stats_output/within_Sim_Boundary_Items.csv')
    















def checkCBSBetween_BoundaryItems(cold, fullClass, reducedClass, noObs):
    
    
   #Get the data for the cold similarity ratings
    print('cold sim descriptives:')
    dfCold = pd.read_csv(cold)
    dfCold = dfCold.replace(' Beta', 'Beta')

    dfColdAB = dfCold.drop(index = dfCold[dfCold['stimId1'] != 'A450'].index)
    dfColdAB = dfColdAB.drop(index = dfColdAB[dfColdAB['stimID2'] != 'B550'].index)

    dfColdBG = dfCold.drop(index = dfCold[dfCold['stimId1'] != 'B950'].index)
    dfColdBG = dfColdBG.drop(index = dfColdBG[dfColdBG['stimID2'] != 'C1050'].index)

   

    
    coldMeans= [dfColdAB['response'].mean(),dfColdBG['response'].mean()]
    coldSD = [dfColdAB['response'].std(),dfColdBG['response'].std()]



    #get the data for the full classification condition similarity ratings
    print('full calss sim descriptives:')
    dfFull = pd.read_csv(fullClass)

    dfFull = dfFull.replace(' Beta', 'Beta')
  

    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1350'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimId1'] == 'C1450'].index)
    dfFull = dfFull.drop(index = dfFull[dfFull['stimID2'] == 'C1450'].index)

    dfFullAB = dfFull.drop(index = dfFull[dfFull['stimId1'] != 'A450'].index)
    dfFullAB = dfFullAB.drop(index = dfFullAB[dfFullAB['stimID2'] != 'B550'].index)



    dfFullBG = dfFull.drop(index = dfFull[dfFull['stimId1'] != 'B950'].index)
    dfFullBG = dfFullBG.drop(index = dfFullBG[dfFullBG['stimID2'] != 'C1050'].index)


    FullMeans= [dfFullAB['response'].mean(),dfFullBG['response'].mean()]
    FullSD = [dfFullAB['response'].std(),dfFullBG['response'].std()]







    #get the data for the reduced classification condition similarity ratings
    print('reduced calss sim descriptives:')
    dfReduced = pd.read_csv(reducedClass)

    dfReduced = dfReduced.replace(' Beta', 'Beta')




    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1350'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] == 'C1450'].index)
    dfReduced = dfReduced.drop(index = dfReduced[dfReduced['stimID2'] == 'C1450'].index)

    dfReducedAB = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] != 'A450'].index)
    dfReducedAB = dfReducedAB.drop(index = dfReducedAB[dfReducedAB['stimID2'] != 'B550'].index)

    dfReducedBG = dfReduced.drop(index = dfReduced[dfReduced['stimId1'] != 'B950'].index)
    dfReducedBG = dfReducedBG.drop(index = dfReducedBG[dfReducedBG['stimID2'] != 'C1050'].index)



    ReducedMeans= [dfReducedAB['response'].mean(),dfReducedBG['response'].mean()]
    ReducedSD = [dfReducedAB['response'].std(),dfReducedBG['response'].std()]





    #get the data for the no observation condition similarity data
    print('no obs calss sim descriptives:')
    dfNoObs = pd.read_csv(noObs)

    dfNoObs = dfNoObs.replace(' Beta', 'Beta')


    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1350'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] == 'C1450'].index)
    dfNoObs = dfNoObs.drop(index = dfNoObs[dfNoObs['stimID2'] == 'C1450'].index)


    dfNoObsAB = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] != 'A450'].index)
    dfNoObsAB = dfNoObsAB.drop(index = dfNoObsAB[dfNoObsAB['stimID2'] != 'B550'].index)

    dfNoObsBG = dfNoObs.drop(index = dfNoObs[dfNoObs['stimId1'] != 'B950'].index)
    dfNoObsBG = dfNoObsBG.drop(index = dfNoObsBG[dfNoObsBG['stimID2'] != 'C1050'].index)




    NoObsMeans= [dfNoObsAB['response'].mean(), dfNoObsBG['response'].mean()]
    NoObsSD = [dfNoObsAB['response'].std(), dfNoObsBG['response'].std()]






    numsubs = len(dfCold['id'].unique()) + len(dfFull['id'].unique()) +len(dfReduced['id'].unique()) + len(dfNoObs['id'].unique())

    #quick anaova with scipy because more simple to set up than the stats models ols
    print('ANOVAs')
    ABstat, ABpval= f_oneway(dfColdAB['response'], dfFullAB['response'], dfReducedAB['response'], dfNoObsAB['response'])

    BGstat, BGpval= f_oneway(dfColdBG['response'], dfFullBG['response'], dfReducedBG['response'], dfNoObsBG['response'])
    
    

    output = pd.DataFrame(
        {
            'Category-Diff' : ['Alpha-Beta', 'Beta-Gamma',],
            'Cold Means': coldMeans,
            'Cold SD' : coldSD,
            'Full Means' : FullMeans,
            'Full SD' : FullSD,
            'Reduced Means' : ReducedMeans,
            'Reduced SD' : ReducedSD,
            'No Obs Means' : NoObsMeans,
            'No Obs SD' : NoObsSD,
            'F statistic' : [ABstat, BGstat],
            'p value' : [ABpval, BGpval],
            'n': [numsubs,numsubs]
        },
    )
    print(output)
    output.to_csv('Analysis/stats_output/Between_Sim_Boundary_Items.csv')
    





'''
cold= 'cold_sim_results.csv'
fullClass= 'cond_2_similarity_results.csv'
reducedClass = 'cond_1_similarity_results.csv'
noObs= 'cond_3_similarity_results.csv'
checkCBSBetween_BoundaryItems(
    cold=cold,
    fullClass=fullClass,
    reducedClass=reducedClass,
    noObs=noObs
    )'''