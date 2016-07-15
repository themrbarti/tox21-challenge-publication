# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:09:24 2014

@author: gergo
"""


import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

#==============================================================================
# Data prepartation steps
#==============================================================================

dset = 'nr-aromatase'

dfap = pd.read_csv('allin-rca-lmv.csv',sep=';',na_values=[''], dtype=np.object)

# Remove columns w/ excessive missing data
for i in dfap.columns:
    #if (pd.isnull(dfap[i]).sum() >= len(dfap)*0.1):
    if (pd.isnull(dfap[i]).sum() > 225):
        print('Deleting ',i, pd.isnull(dfap[i]).sum())
        del dfap[i]

# Discard non-numerical data, conversion
mols = dfap['Molecule name']        
del dfap['Molecule name']        
del dfap['Salt Stripped Molecule']        
dfap = dfap.astype(np.float)
dfap['Molecule name']  = mols  


#==============================================================================
# Construct taining set
#==============================================================================
dfs = pd.read_csv('all-molecules.csv',sep=',',na_values=[''], dtype=np.object)

# sanitize column names
dfap.columns = [col.lower().strip().replace(' ','_') for col in dfap.columns.values]
dfs.columns = [col.lower().strip().replace(' ','_') for col in dfs.columns.values]

# Select proper target attr
dfs['active'] = dfs[dset]

    
train = pd.merge(dfap, dfs[['molecule_name','active']].dropna(), on = 'molecule_name', how = 'inner', suffixes = ['_l', '_r'])   



#==============================================================================
# Construct test set
#==============================================================================
test = pd.read_csv('allfinal.csv',sep=';',na_values=[''], dtype=np.object)
test.columns = [col.lower().strip().replace(' ','_') for col in test.columns.values]


test = test[train.columns.difference(['active'])]

#==============================================================================
# Handle missing data
#==============================================================================
test = test.fillna(0)
train = train.dropna()

#==============================================================================
# Clean up tables
#==============================================================================
# Discard non-numerical data, conversion
del train['molecule_name']
molecule_name = test['molecule_name']
del test['molecule_name']

               
test = test.astype(np.float)
            
y = train['active']            
del train['active']

y = y.astype(np.int)

#==============================================================================
# Filter attributes by RapidMiner output
#==============================================================================
tr_weights = pd.read_csv('weights-relief-03.csv', sep = ';',na_values=[''])   
tr_weights['Attribute'] = tr_weights.Attribute.apply(lambda x: x.lower())
tr_weights = tr_weights.sort_values(by = ['Total_Weight'], ascending=[False])
cols =  list(tr_weights[tr_weights['Total_Weight'] > 0.3].ix[:,'Attribute'])

#==============================================================================
# Modeling
#==============================================================================
# Define model
mod = ExtraTreesClassifier(n_estimators = 999, criterion = 'entropy')


# Train model
mod.fit(train[cols],y)

# Extract top200 feature importances
imps = pd.Series(mod.feature_importances_, index=cols)
imps.sort_values(ascending=False, inplace=True)
impd = pd.DataFrame(imps.head(200)).reset_index()
impd.columns = ['Feature','Relative Importance']
impd.to_csv('feature_importances_%s.csv' % (dset),sep=',',index=False)

# Issue predictions
y_pred = mod.predict(test[cols])
y_proba = mod.predict_proba(test[cols])

subm = pd.DataFrame([x[1] for x in y_proba.tolist()],columns = ['proba'])

# Select threshold based on training distribution
thr = 0.45

subm['active'] = subm.proba.apply(lambda x: 1 if x>=thr else 0)
subm['molecule_name'] = molecule_name



subm.rename(columns={'molecule_name' : 'Sample ID','proba' : 'Score','active' : 'Activity'}, inplace=True)

subm[['Sample ID','Score','Activity']].to_csv(dset +'-ef.csv',sep='\t',index=False)