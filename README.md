# A-PREDICTIVE-ANALYSIS-APPROACH-USING-LINEAR-REGRESSION-TO-ESTIMATE-SOFTWARE-EFFORT
ABSTRACT.          Software engineering projects typically have a higher level of uncertainty and its very difficult to accurately estimate the time required to do work when there is a high level of uncertainty associated with the work to be done.        Predicting software quality requires high accurate tools and high-level experience. AI-based predictive models, on the other hand, are useful tools with an accurate degree that help to make decisions learning from past data. In this study, to build a software effort estimation model to predict the effort before the project development lifecycle, using a linear regression model and also using non-parametric validation model through a Knn regression algorithm.
1.	INTRODUCTION
  Software development involves a number of interrelated factors which affect development effort and productivity . 
An optimal software development process is regarded as being dependent on the situational characteristics of individual software development settings. Such characteristics include the nature of the application(s) under development, team size, requirements volatility and personnel experience. At the highest level, a software engineer is responsible for researching, designing, implementing, and supporting software solutions to various problems. The most significant activity in software engineer is the development of projects within the confined timeframe and budget,  So accuracy has a vital role for software development, effort prediction estimation is one of the critical tasks required for developing software. In this report focus on analyzing the importance of attributes in estimating software cost as well as its correlation.


In this report, we set out to answer two research questions related to the dataset:
I.	Which the correlation of each metrics in the estimation of software effort ?
II.	How accurate is the model of software effort ?
2.	SOFTWARE EFFORT ESTIMATION
Effort estimation is the process of forecasting how much effort is required to develop or maintain a software application. This effort is traditionally measured in the hours worked by a person, or the money needed to pay for this work. 
               Effort estimation is used to help draft project plans and budgets in the early stages of the software development life cycle. This practice enables a project manager or product owner to accurately predict costs and allocate resources accordingly.
When measurements embrace structure system they become more meaningful indicators called metrics. Metrics are conceived by the user and designed to reveal chosen characteristics in a reliable meaningful manner. Then these metrics are mapped to ongoing measurements, to arrive at a best fit [1].
One of the fundamental issues in a software project is to know, before executing it, how much effort, in working hours, it will be necessary to bring it to term. This area called effort estimation counts on some techniques that have presented interesting results over the last few years [2].
One of reasons for failed estimations is an insufficient background of information in the area of software estimation. Unfortunately, human experts are not always as good at estimating as one could hope: estimates of cost and effort in software projects are often inaccurate, with an average overrun of about 30% [3].
Deliberate decisions regarding the particular estimation method and knowledgeable use require insight into the principles of effort estimation [4, p. 2014].
Learning-oriented models attempt to automate the estimation process by building computerised models that can learn from previous estimation experience [5].These models do not rely on assumptions and are capable of learning incrementally as new data are provided over time [6].
2.1.	RELATED WORKS
From the research developed by Ayyıldız makes use of Desharnais dataset to finding the necessary attributes that affects the software effort estimation and analyzing the necessity of these attributes [6]. The Pearson’s Correlation correlations between metrics of Desharnais dataset and software effort are analyzed and applicability of the regression analysis is examined.
To show the differences between the actual and estimated values of the dependent variable, prediction performance are evaluated using Root Squared, Mean Squared Error and Root Mean Squared Error.
3.	MATERIALS AND METHODS
To perform our study firstly we analyze the correlation between each attributes of Desharnais dataset and effort attribute. We apply linear regression technique to investigate relation between these attributes. After that we apply a regression based on k-nearest neighbors regressor. Lastly we evaluate our prediction performance comparing the squared error value of both algorithms .
3.1.	DATASET
To perform this study we used Desharnais dataset which is composed of a total of 81 projects developed by a Canadian software house in 1989. This data set includes nine numerical attributes. The eight independent attribute of this data set, namely ”TeamExp”, ”ManagerExp”, ”YearEnd”, ”Length”, ”Transactions”, ”Entities”, ”PointsAdj”, and ”PointsNonAjust” are all considered for constructing the models. The dependent attribute “Effort” is measured in person hours.
3.2.	FEATURE SELECTION
To address Desharnais dataset the correlations between attributes and software effort are analyzed. The correlation between two variables is a measure of how well the variables are related.A feature is an individual measurable property of the process being observed.
The most common measure of correlation in statistics is the Pearson Correlation Pearson correlation coefficient (PCC), which is a statistical metric that measures the strength and direction of a linear relationship between two random variables [7] Pearson correlation coefficient analysis produces a result between -1 and 1. Results between 0.5 and 1.0 indicate high correlation [8] The Pearson correlation coefficients between attributes and software efforts are given in Figure 1 for Desharnais dataset.
 
Figure 1. Pearsons Correlation to Desharnais dataset
3.3.	MODELS CONSTRUCTION
In this study the following algorithms were used: Linear Regression and K-Nearest Neighbors Regression. The training of the models was carried out in Python language, along with the following libraries: Numpy, Pandas, Scikit-learn, Seaborn and Matplotlib. During the training it was necessary to estimate the values of the random state parameter, since they are not previously known.
Regression analysis aim to explain variability in dependent variable by means of one or more of independent or control variables.. The training of the Linear Regression model consists of generating a regression for the target variable Y. Thus A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0). Likewise the K-Nearest Neighbor Regression is a simple non-parametric method algorithm, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. Non-parametric methods are methods of statistical analysis that do not require a distribution to meet the required assumptions to be analyzed (especially if the data is not normally distributed). Due to this reason, they are sometimes referred to as distribution-free methods. In fact choice of a K-Nearest Neighbor Regression was motivated by the absence of a detailed explanation about how effort attribute value is calculated on Desharnais dataset.
4.	RESULTS
From models generated the training data 70% and 30% testing data for Linear Regression Model and for K-Nearest Neighbor Regression the training data 67% and 33% testing the data , previously isolated, and their performances is evaluated in order to demonstrate how accurate the linear regression model can predict software effort estimation.. Thus,calculating the respective R2 values. Table 3 shows the coefficients reached.
Algorithm	R2 Score
Linear Model Regression	0.768007495444071
K-Nearest Neighbor Regressor	0.7124067273589926
Table 1: Algorithms model results
In Figure 2 plots of the best correlated variables applied to both models are displayed.

  

(a)	Knn x LR on Length feature
 	
	(b) Knn x LR on Transactions feature
 
(b)	Knn x LR on Entities feature
 	
(d) Knn x LR on PointsNonAdjust feature
 
(e) Knn x LR on PointsAjust feature
Figure 2. Comparative R2 scores from K-neighbors Regression and Linear Regression
Each feature from more correlated features is illustrated in Figure 2. The figure shows the linear model (blue line) prediction is fairly close to Knn model effort prediction (green line), predicting the numerical target based on a similarity measure.
5.	CONCLUSION AND EVALUATION
The contributions of this work are based on the use of two output models that seek to take advantage of the relationships between the target values of the project. These methods, together with linear regression and K-neighbors regression algorithms , resulted in predictive models capable of estimating values for the software effort estimation operations. The results of our empirical study reveal that predictive model of software effort presented by both models, could successfully predict more than 70%  by using Linear Regression Model with less than 5% difference between them.
Our results obtained obtained a R2 value of more than 70% and a difference of only 5% among them, indicating the feasibility of using linear regressors to predict software effort. However, to have a more concise and fair result we need to reproduce the same approach with other available algorithms.
Finally, Both happen, both are to be expected, and both are important clues for planning. When the estimates are far apart it signals that the team understands the problem differently. A developer with a low estimate may know of a library or a tool that can speed things up.
References

[1] 		C. R. Pandian, Software metrics: A guide to planning, analysis and application, 2003. 
[2] 		R. (. Wazlawick, ENGENHARIA DE SOFTWARE: CONCEITOS E PRATICAS, Elsevier Editora Ltda, 2013. 
[3] 		Halkjelsvik, T. and Jørgensen, M. (2011)., "From origami to software development: A review of studies on judgment-based predictions of performance time.," p. 138:238–71..
[4] 		Trendowicz, A. and Jeffery, R., Software Project Effort Estimation: Foundations and Best Practice Guidelines for Success. Springer Publishing Company, Incorporated., 2014. 
[5] 		B. A. C. a. C. S. Boehm, "Software development cost estimation approaches &ndash; a survey. Ann. Softw. Eng.," 2000, pp. 10(1-4):177–205..
[6] 		A. C. C. H. a. B. J. Lee-Post, "Software development cost estimation: Integrating neural network with cluster analysis.," 1998, p. 34:1–9..
[7] 		Rodgers, J. and Nicewander, W., "Thirteen ways to look at the correlation coefficient. The American Statistician,," 1988, p. 42(1):59–66..
[8] 		Mehedi Hassan Onik, M., Ahmmed Nobin, S., , in Ferdous Ashrafi, A., and Mohmud Chowdhury, T. (2018). Prediction of a Gene Regulatory Network from Gene Expression Profiles With Linear Regression and Pearson Correlation Coefficient. , ArXiv e-prints.. 
[9] 		Erc¸elebi Ayyıldız, T. and Can Terzi, H. (2017). , "Case study on software effort estimation.," p. 7:103–107..



