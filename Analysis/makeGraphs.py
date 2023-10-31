import analysis_utils as au

learn_curve_paths = {
    'C:/Users/apers/line_data/cond_1_trainPhase_results.csv' : 'Learning Curve Reduced Classification',
    'C:/Users/apers/line_data/cond_2_trainPhase_results.csv' : 'Learning Curve Full Classsification',
    'C:/Users/apers/line_data/cond_3_trainPhase_results.csv' : 'Learning Curve No Observation',
}
full_learn_curve_paths = {
    'C:/Users/apers/line_data/cond_2_trainPhase_results.csv' : 'Learning Curve Full Classsification',
    'C:/Users/apers/line_data/cond_3_trainPhase_results.csv' : 'Learning Curve No Observation',
}
reduced_learn_curve_paths = {
    'C:/Users/apers/line_data/cond_1_trainPhase_results.csv' : 'Learning Curve Reduced Classification',
}
two_gen_paths = {
    'C:/Users/apers/line_data/cond_2_testPhase_results.csv' : 'Generalization Full Classification',
    'C:/Users/apers/line_data/cond_3_testPhase_results.csv' : 'Generalization No Observation',
}
three_gen_paths = {
    'C:/Users/apers/line_data/cond_1_testPhase_results.csv' : 'Generalization Reduced Classification',
}
typ_paths = {
    'C:/Users/apers/line_data/cond_1_typicality1_results.csv' : 'Pre-Classification Typicality Reduced Classification',
    'C:/Users/apers/line_data/cond_1_typicality2_results.csv' : 'Post-Classification Typicality Reduced Classification',
    'C:/Users/apers/line_data/cond_2_typicality1_results.csv' : 'Pre-Classification Typicality Full Classification',
    'C:/Users/apers/line_data/cond_2_typicality2_results.csv' : 'Post-Classification Typicality Full Classification',
    'C:/Users/apers/line_data/cond_3_typicality2_results.csv' : 'No Observation',

}
test_accs = {
    'C:/Users/apers/line_data/cond_2_testPhase_results.csv' : 'Test Accuracy Full Classification',
    'C:/Users/apers/line_data/cond_3_testPhase_results.csv' : 'Test Accuracy No Observation',
    'C:/Users/apers/line_data/cond_1_testPhase_results.csv' : 'Test Accuracy Reduced Classification',

}
accHard_gens = {
    'C:/Users/apers/line_data/cond_2_testPhase_results.csv' : 'Generalization Full Classification',
    'C:/Users/apers/line_data/cond_3_testPhase_results.csv' : 'Generalization No Observation',
    'C:/Users/apers/line_data/cond_1_testPhase_results.csv' : 'Generalization Reduced Classification',

}

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


#get overall learn curves
for key in learn_curve_paths:
    au.plotLearnCurve(path= key, title= learn_curve_paths[key])


#get per cat and per item learn curves for three-way classification
for key in full_learn_curve_paths:
    au.fullPerCatLearnCurves(path= key, title= full_learn_curve_paths[key], changestimID=True, changes=changes)



#get per cat and per item learn curves for three-way classification
for key in reduced_learn_curve_paths:
    au.reducedPerCatLearnCurves(path= key, title= reduced_learn_curve_paths[key], changestimID=True, changes=changes)



#get typicality
for key in typ_paths:
   au.plotTyps(path=key, title= typ_paths[key], changestimID=True,changes=changes)

#get test accuracy for training items
for key in test_accs:
   au.getTestAcc(path=key, title= test_accs[key], changestimID=True,changes=changes)

#get generalization performance when subjects only ever gave two responses
for key in two_gen_paths:
    au.twoGetGens(path=key, title=two_gen_paths[key])



#get generalization behavior when subjects gave all three responses at some point
for key in three_gen_paths:
    au.threeGetGens(path=key, title=three_gen_paths[key])

#get gens for ss who answered the hard items correctly
for key in accHard_gens:
    au.HardTestRight_GetGens(path=key, title=accHard_gens[key])