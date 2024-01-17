Model Monitoring and Data Drift
-------------------------------

1. Covariance Drift : Variance in Independent Variable 
2. Concept Drift: Variance in Target Variable

ML Testing is Highly Complex
1. AI follows a data-driven programming paradigm : 3 Dimension influence the code, the model, and the data
2. AI is not easily breakable in small unit components : It is hard to break the AI system into smaller components - training data, the learning program, and the learning library
3. AI errors are systemic and self-amplifying : output of one model can be ingested into the training base of another.AI errors can be difficult to identify, measure, and correct.

Testing Methods in ML
----------------------

Name | Description
:- | :-
Behavioral testing | metamorphic testing, heuristics testing
Drift testing      | Kulback divergence, Kolmogorov-Smirnov , Population stability index (PSI)
Performance testing| model error testing, calibration score, simple model comparison


<B> Types of Drift </B>
------------------------
1. Concept Drift  : Distribution changes in Target Variable.
2. Covariance Drift : Changes in Distribution of Independent Variables. Ex : Salary can increase from 1000-2000 to 10000-20000

<B>Methods for Detecting Data Drift</B>
------------------------------------------------
1. Kolmogorov-Smirnov (K-S) test:
   Compares Data DIstribution of 2 datasets. The null hypothesis for this test states that the data distributions from both the datasets are same.
   If the null is rejected then we can conclude that there is adrift in the model.
   
<img width="216" alt="image" src="https://github.com/klnsuman/Dissertation/assets/11458777/aa955d8a-2fc5-4ed3-acc5-14eab89309f4">

2. Population Stability Index: PSI
   It compares the distribution of the target variable in the test dataset to a training data set that was used to develop the model.
   <img width="507" alt="image" src="https://github.com/klnsuman/Dissertation/assets/11458777/97365fe1-559e-4fde-8701-6f4625cf47f0">

   a) When PSI<=1
    This means there is no change or shift in the distributions of both datasets.
    
    b) 0.1< PSI<0.2
    
    This indicates a slight change or shift has occurred.
    
    c) PSI>0.2
    
    This indicates a large shift in the distribution has occurred between both datasets.

   https://www.analyticsvidhya.com/blog/2021/10/mlops-and-the-importance-of-data-drift-detection/


