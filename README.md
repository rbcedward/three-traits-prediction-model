# Three facial traits (Trustworthness, Dominance, Attractiveness) prediction flow
The picture of CEO contains lots of information. When seeing photo, we make some judgement about the person in it. In order to quantify what people see and feel. We conduct a model to train and predict the htree facial traits , Trustworthness, Dominance and Attractiveness of a CEO. In these article, the steps of how we predict those traits are not shown here. Only offer the trained model by python. The main target is to show how to use it to get the three traits of the photo user collected. First, we have to say there is a constraint, that the number of photos should be greater than 30 when using these code and modes. 

## Data and Photo
A data set of list of CEO is required in the begining, here is the example of excel file that must contain these column

![image](https://user-images.githubusercontent.com/37869717/161537080-07451670-b0d9-4124-a2e2-2eba00ab09f5.png)

"TICKER" is the ticker of a company, "GLASS","BEARD","STUBBLE" are indicator of whether the person in picture is wearing a glasses, have beard and have stubble. These should be manually conduct when collecting photo, these columns are what will be used in next step of stages.

## Python code 
### preamble
After collecting data and noted those three columns, we can flip to next part, which is python code. In this article, we use a lot of modules and packages, The  preamble.ipynb contains code of those modules. If user have not downloaded before, please take some time to finishing downlaoding or updateing those modules and packages first.

### functions of calculating of photo
In this part, we define lots of functions, some of it is refer some code from others, but I can't remember where I found these. If user know the source, please tell me, that I could add the reference of those code written by others.
