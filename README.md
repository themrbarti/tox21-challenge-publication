# Tox21 Data Challenge publication supplementary material
Supplementary materials for Tox21 Data Challenge solution of team Dmlab. Companion to the article "Identifying biological pathway interrupting toxins using multi-tree ensembles" published in the special issue of Frontiers in Environmental Science journal.

The publication is currently under review.

Materials included here:
 - /configuration/padel-descriptors.conf: Configuration file to generate descriptors & fingerprints in PaDel
 - /misc/feature_importances_nr-aromatase.csv: top200 most important features by the random forest model for winning track NR-aromatase
 - /misc/feature_importances_nr-ar.csv: top200 most important features by the random forest model for winning track NR-AR
 - /misc/feature_importances_sr-p53.csv: top200 most important features by the random forest model for winning track SR-p53
 - /misc/weights-relief-03.csv: Attribute weights generated by the attribute weighting scheme in RapidMiner
 - /processes/python/nr-aromatase.py: sample data preparation & modeling for winning track NR-aromatase
 - /processes/python/nr-ar.py: sample data preparation & modeling for winning track NR-AR
 - /processes/python/sr-p53.py: sample data preparation & modeling for winning track SR-p53
 - /processes/rapidminer/4-preparing-final-evaluation-set.rmp: RapidMiner process to generate final evaluation data set
 - /processes/rapidminer/3-selecting-attributes-by-weights.rmp: RapidMiner process for attribute weighting scheme
 - /processes/rapidminer/2-remove-missing-values.rmp: RapidMiner process to handle missing data
 - /processes/rapidminer/1-prepare-descriptions-fingerprints.rmp: RapidMiner process to prepare data set
 - /processes/knime/2-fingerprint-generation.zip: KNIME process to generate fingeprints
 - /processes/knime/1-descriptor-generation.zip: KNIME process to generate descriptors

The solution was developed using the following software versions:

PaDel Descriptor
 - available from http://www.yapcwsoft.com/dd/padeldescriptor/

KNIME Analytics Platform 2.10.1
 - available from http://www.knime.org/downloads/overview

RDKit KNIME Extension 2.4.0
 - available from http://tech.knime.org/community/rdkit and the KNIME marketplace

RapidMiner 5.3.15
 - available from http://sourceforge.net/projects/rapidminer/files/1.%20RapidMiner/5.3/

RapidMiner Feature Selection Extension 1.1.4
 - available from http://sourceforge.net/projects/rm-featselext/ and the RapidMiner marketplace

Python 2.7.5
 - available from https://www.python.org/downloads/

Pandas library 0.14.1
 - available from http://www.lfd.uci.edu/~gohlke/pythonlibs/

Scikit-learn library 0.15.0
 - available from http://www.lfd.uci.edu/~gohlke/pythonlibs/
