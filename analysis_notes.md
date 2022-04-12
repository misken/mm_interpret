OB metamodel paper analysis notes
==================================

Exp11 - small scenario space
-----------------------------

The original simulation mm experiment involved 135 scenarios:

- 5 arrival rate volumes
- 3 c-section probabilites
- 3 ldr capacity targets
- 3 pp capacity targets

Metamodels were fit for OBS, LDR, and PP for mean occupancy, 95th percentile
of occupancy, blocking probabilities and conditional mean blocking time.

Five-fold cross validation was used to fit and assess models.

Actual vs Predicted plots were created for all models. These were based
one prediciton for each scenario associated with the fold in which that
scenario was in the test dataset. So, plots represent performance on
test data within the experimental design.

Need a tabular summary of all models, all units, all performance measures
which incudes both MAE and MAPE

* In mm_interpret/output you can find exp11_{unit}_metrics_df.csv
* Each table has one row per fold for each combination of:
    - unit
    - performance measure
    - feature set (noq, q, basicq, onlyq)
    - model
* The metrics include 
    - MAE on test and train
    - MAPE on test and train
    - R^2 on test and train
    
Need to aggregate the metrics over the folds within the unit_pm_qdata_model value.
