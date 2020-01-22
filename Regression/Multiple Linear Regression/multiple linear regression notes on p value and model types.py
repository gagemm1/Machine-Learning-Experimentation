# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:16:29 2019

@author: gagmorri

mutliple linear regression is similar to linear, and the equation looks still a lot like y = mx +b, just with more m's
something like y = mx + nx + ox + b

in this example, we have our independent variable (profit) with dependents R&D spending, admin, marketing, and state. 

State in this case is a categorical variable (it's a word) so we need to change the states to dummy variables just like we did before
but since there are are only two categories we can use only 1 dummy variable
this is key because if we used both of them (and as a binary, NY = 1 - California or vise versa)
if we included both, that would mean we're pretty much duplicating a variable because you can directly know the value of 1 given the other
this is called the dummy variale trap. Always exclude 1 of your dummy variables for each category/dummy set (like state is 1 dummy variable and industry is the other)
the dummy variable trap is also called multicollinearity

the p (probability) value tells you how likely it is to get a result like you have if the null hypothesis is true
so we're assuming the null hypothesis is true and then determine how "strange" our sample is. If it is a large p value (not "strange") then that supports the idea that
the null hypothesis is true (the one we're trying to disprove). So that means if we're trying to disprove our null hypothesis, we want our p value to be very small, 
and that's why in the model we reject the items with the largest p value and anything below our significance level is good.

¡¡So, to sum up, large p values are bad, low p values are good because they are supporting disprove what you want to disprove.!!
Usually we're saying x independent variable has no correlation with y dependent variable, and if P is high then that's likely to be true


Model selection:
In the backwards elimination model building we will reject variables in the model until the predictor (values/columns)
with the highest p value are still below the significance level (most often 5%)

In the forward selection model we do pretty much the same stuff except forward and keep adding variables and stop when you have a P > SL and choose the previous model 

bi-directional: refer to pdf

"""

 