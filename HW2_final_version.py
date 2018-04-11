
# coding: utf-8

# first, I need to import the necessart libraries for now; I will be adding more later on as I need them. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# now, I need to import my dataframe into this file.

# In[2]:


my_data = pd.read_csv("IBM_people_dataset.csv")


# In[4]:


pd.set_option('display.max_columns', None) #because I need to see the whole data


# In[5]:


my_data.head()


# In[6]:


my_data.info()


# We do not have any mssing data, since all of the categories have 1470 observations. However, I can see that some of the data are categorical (object), so I need to turn them into numerical. I can also see that there are some variables that I do not need obviously in my algorithm, so I will try to understand better which variables are useless and remove them from the dataframe. First let me see what are the categorical variables composed of so that I can understand and analyze them further. 

# In[7]:


my_data.Attrition.unique()


# In[8]:


my_data.BusinessTravel.unique() #can be turned into an ordinal numerical variable


# In[9]:


my_data.Department.unique()


# In[10]:


my_data.EducationField.unique()


# In[11]:


my_data.Gender.unique()


# In[12]:


my_data.JobRole.unique()


# In[13]:


my_data.MaritalStatus.unique()


# In[14]:


my_data.Over18.unique()


# In[15]:


my_data.OverTime.unique()


# In[16]:


my_data = my_data.drop("Over18", axis=1) #since it has the same value for everyone


# In[17]:


my_data = my_data.drop("JobRole", axis=1) #since it is extremely similar to and shows the same thing as "Department"


# In[18]:


my_data = my_data.drop("EmployeeNumber", axis=1) #since it is unique for every obersvation and is useless since it shows a random number


# In[19]:


my_data.EmployeeCount.unique()


# In[20]:


my_data = my_data.drop("EmployeeCount", axis=1) #since again it only has the one same value for every observation and is completely useless


# In[22]:


my_data.Education.unique() #not dropping this


# In[23]:


my_data.StandardHours.unique() #of course dropping this for same reasons mentioned above


# In[24]:


my_data = my_data.drop("StandardHours", axis=1)


# In[28]:


my_data.EnvironmentSatisfaction.unique() #not dropping this either, just observing what it is


# now I will turn all the categorical variables into numerical

# In[29]:


my_data.info()  #just remembering what I have left


# In[30]:


Churn = pd.get_dummies(my_data.Attrition)


# In[31]:


my_data = my_data.drop("Attrition", axis=1)


# In[32]:


Churn = Churn.drop("No", axis=1)  #now 1s are churners, 0s are non-churners


# In[33]:


my_data = my_data.join(Churn) #called Yes, will rename later


# In[34]:


my_data.BusinessTravel = my_data.BusinessTravel.astype("category").cat.reorder_categories(["Non-Travel","Travel_Rarely","Travel_Frequently"]).cat.codes


# In[35]:


Department = pd.get_dummies(my_data.Department)


# In[36]:


my_data = my_data.drop("Department", axis=1)


# In[37]:


Department = Department.drop("Human Resources", axis=1) #due to DoF


# In[38]:


my_data=my_data.join(Department)


# In[39]:


ed_field = pd.get_dummies(my_data.EducationField)


# In[40]:


my_data = my_data.drop("EducationField", axis=1)


# In[41]:


ed_field =ed_field.drop("Other", axis=1)


# In[42]:


my_data = my_data.join(ed_field)


# In[43]:


gender = pd.get_dummies(my_data.Gender)


# In[44]:


my_data = my_data.drop("Gender", axis=1)


# In[45]:


gender = gender.drop("Male", axis=1) #now 1s are female, 0s are male


# In[46]:


my_data = my_data.join(gender) #now it's called "Female" but i'll change the name later


# In[47]:


marital_status = pd.get_dummies(my_data.MaritalStatus)


# In[48]:


my_data = my_data.drop("MaritalStatus", axis=1)


# In[49]:


marital_status = marital_status.drop("Divorced", axis=1)


# In[50]:


my_data = my_data.join(marital_status)


# In[51]:


overtime = pd.get_dummies(my_data.OverTime)


# In[52]:


my_data = my_data.drop("OverTime", axis=1)


# In[53]:


ovrtime = overtime.drop("No", axis=1) #now 1s are overtime, 0s are not


# before adding overtime to my data, i'll change the names so that the "Yes" of overtime doesn't overlap with the "yes" of churn

# In[54]:


my_data=my_data.rename(columns = {"Female" : "Gender", "Yes" : "Churn", })


# In[55]:


overtime = overtime.rename(columns = {"Yes" : "Overtime"})


# In[56]:


my_data = my_data.join(overtime)


# now let me make sure everything is ok with my data

# In[57]:


my_data.info()


# I have no idea what No is

# In[58]:


my_data.head()


# ok so No is the opposite of Overtime, I guess it wasn't dropped. No problem.

# In[59]:


my_data = my_data.drop("No", axis=1)


# In[60]:


my_data.info()


# ok everything seems fine except for the fact that there are a lot of variables in the dataframe, so I will try to understand which variables are highly correlated so that I can remove them from the model

# In[62]:


correlation_1 = my_data.corr(method = "pearson")


# In[63]:


correlation_1


# MonthlyIncome and JobLevel are highly correlated (>0.9), however, monthly income is less correlated with churn so it will the one removed from the model

# In[64]:


my_data = my_data.drop("MonthlyIncome", axis=1)


# In[65]:


correlation_2 = my_data.corr(method = "pearson")


# In[66]:


correlation_2


# sales and research & developement are also highly correlated (>0.9), however, sales will be removed since it is less correlated with churn

# In[67]:


my_data = my_data.drop("Sales", axis=1)


# In[68]:


correlation_3 = my_data.corr(method="pearson")


# In[69]:


correlation_3


# performance rating and percent salary hike are highly correlated, I will remover performance rating since it is less correlated with churn

# In[70]:


my_data = my_data.drop("PerformanceRating", axis=1)


# In[71]:


correlation_4 = my_data.corr(method="pearson")


# In[72]:


correlation_4


# there are some variables left that are also correlated, but they have relatively high correlations with churn so i's rather not remove them

# now let's try to create the algorithms. 

# In[74]:


ibm_logit = LogisticRegression(random_state=42)
ibm_tree = DecisionTreeClassifier(random_state=42)


# In[75]:


y = my_data.Churn #what we want to predict
x= my_data.drop("Churn", axis=1)


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[78]:


ibm_logit.fit(x_train,y_train)


# In[79]:


ibm_logit.score(x_train,y_train)*100


# In[80]:


ibm_logit.score(x_test,y_test)*100


# low accuracy, yet no (or maybe not a lot because idk what it means when overfitting is higher than accuracy) overfitting. I will try to get higher accuracy though. 

# In[82]:


ibm_tree.fit(x_train,y_train)


# In[83]:


ibm_tree.score(x_train,y_train)*100


# In[84]:


ibm_tree.score(x_test,y_test)*100 


# as we can see, there is high overfitting in this case. 

# we can use some tools to fight this overfitting problem such as: limiting the tree's growth, implementing a minimum sample size and so on. we should also see if we have a big disbalance between churners and non-churners (probably we do) and balance them to get a higher recall score. 

# In[85]:


100.0*my_data.Churn.value_counts()/len(my_data)


# that classifies as big disbalance, so I will keep that in mind

# In[86]:


ibm_tree_7 = DecisionTreeClassifier(max_depth = 7, random_state=42)
ibm_tree_7.fit(x_train, y_train)
ibm_tree_7.score(x_test, y_test)*100


# In[88]:


ibm_tree_7.score(x_train, y_train)*100 #forgot this in the beginning


# less accurate, still there is overfitting (not as bad as the first one)

# In[89]:


ibm_tree_8 = DecisionTreeClassifier(max_depth = 8, random_state=42)
ibm_tree_8.fit(x_train, y_train)
ibm_tree_8.score(x_train, y_train)*100


# In[136]:


ibm_tree_8.score(x_test, y_test)*100#more accurate yet more overfitting


# In[181]:


ibm_tree_leaf = DecisionTreeClassifier(min_samples_leaf=40, max_depth = 8, random_state=42)
ibm_tree_leaf.fit(x_train, y_train)
ibm_tree_leaf.score(x_train, y_train)*100


# In[182]:


ibm_tree_leaf.score(x_test, y_test)*100


# tested many variations of the last one (ibm_tree_leaf), that's the best one I could come up with

# In[102]:


ibm_tree_8_b = DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42)
ibm_tree_8_b.fit(x_train, y_train)
ibm_tree_8_b.score(x_train, y_train)*100


# In[103]:


ibm_tree_8_b.score(x_test, y_test)


# pretttttty bad

# In[104]:


from sklearn.model_selection import cross_val_score


# In[105]:


np.mean(cross_val_score(ibm_tree_8, x, y, cv=10))


# In[106]:


np.mean(cross_val_score(ibm_tree_7, x, y, cv=10))


# In[107]:


from sklearn.metrics import recall_score


# In[108]:


prediction_7 = ibm_tree_7.predict(x_test)


# In[110]:


recall_score(y_test,prediction_7)*100


# In[112]:


prediction_8 = ibm_tree_8.predict(x_test)
recall_score(y_test, prediction_8)*100


# In[113]:


prediction_logit = ibm_logit.predict(x_test)
recall_score(y_test, prediction_logit)*100


# much better recall score, but i'm still not satisfied with this result. I will try to implement a loop to find which parameters will give me the best model in case of decision trees, maybe I will find better models (basically what we learned in class today). 

# In[114]:


from sklearn.model_selection import GridSearchCV


# In[115]:


ibm_test = DecisionTreeClassifier(class_weight="balanced", random_state=42)


# In[116]:


max_depth_values = [i for i in range(5,21)]
min_sample_values = [i for i in range(10,501,10)]


# In[118]:


parameters={"max_depth":max_depth_values,"min_samples_leaf": min_sample_values}


# In[119]:


finder = GridSearchCV(ibm_test, parameters)


# In[120]:


finder.fit(x_train, y_train)


# In[121]:


print(finder.best_params_)


# In[132]:


ibm_tree_best = DecisionTreeClassifier(max_depth = 5, min_samples_leaf=370, class_weight = "balanced", random_state=42)
ibm_tree_best.fit(x_train, y_train)
ibm_tree_best.score(x_train, y_train)*100


# In[133]:


ibm_tree_best.score(x_test, y_test)*100


# basically this model is not good at all because the accuracy is very low. I tried other parameters, still nothing works.

# In[134]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,prediction_logit)*100


# In[135]:


roc_auc_score(y_test, prediction_8)*100


# In[183]:


prediction_leaf = ibm_tree_leaf.predict(x_test)


# In[184]:


recall_score(y_test, prediction_leaf)*100


# In[186]:


roc_auc_score(y_test, prediction_leaf)*100


# Conclusion from all this: the logistic regression model serves as the best algorithm that i could find to predict churn; it has no overfitting problems, higher recall and roc_auc scores than the other models and an acceptable accuracy. 
