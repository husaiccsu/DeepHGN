DeepHGN:
This repository contains the code and datasets for the DeepHGN method, as described in the manuscript.

Quick Start
1. Install Dependencies
   The code was developed and tested using python 3.10    
   pip install networkx
   pip install numpy
   pip install matplotlib
   pip install scikit-learn
   pip install joblib
   pip install seaborn

   
2 Download Data Files:
   Place the following files in './data':
   [Species]_PPI_StringV12.txt (e.g., Homo_sapiens_PPI_StringV12.txt).
   [Species]_Sequence_Uniprot.txt, [Species]__CAFA3.txt.
   Total_DAG.txt (GO hierarchy).
   Species: 'Homo_sapiens' 或 'Saccharomyces_cerevisiae'
   GOType: 'F'（MF）、'P'（BP）、'C'（CC）

3 Run Prediction:
  python DeepHGN.py
  Modify Species and GOType in DeepHGN.py to switch between species (Human/Yeast) and ontologies (BP/MF/CC).
 

Custom Data Support
1.DeepHGN.py: The DeepHGN method
2. Homo_sapiens_CAFA3.txt: The third Critical Assessment of Protein Function Annotation (CAFA3) dataset for Human
3. Homo_sapiens_PPI_StringV12.txt: String PPI network for Human
4. Homo_sapiens_Sequence_Uniprot.txt: The sequence data for Human
5. Saccharomyces_cerevisiae_CAFA3.txt: The third Critical Assessment of Protein Function Annotation (CAFA3) dataset for Yeast
6. Saccharomyces_cerevisiae_PPI_StringV12.txt: String PPI network for Yeast
7. Saccharomyces_cerevisiae_Sequence_Uniprot.txt: The sequence data for Yeast
8. Total_DAG.txt: GO Ontology including Human and Yeast

License
  DeepHGN is released under the MIT License. Contact bihaizhao@163.com for questions.