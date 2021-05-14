# Gaussian Naive Bayes Classifier
**author  : s.aparajith@live.com**  
**date : 14/5/2021**  

---
[formula1]: ./doc/formula0.JPG "iprob"
[formula2]: ./doc/formula01.JPG "iprob1"
[formula3]: ./doc/formula1.JPG "iprob2"
[formula4]: ./doc/formula2.JPG "iprob3"
[formula5]: ./doc/formula3.JPG "iprob3"
[formula6]: ./doc/formula4.JPG "iprob3"
[formula7]: ./doc/formula5.JPG "iprob3"
[formula8]: ./doc/prediction1.JPG "iprob3"
[formula9]: ./doc/prediction2.JPG "iprob3"
[data1]: ./doc/dat.png "ata"

## Building the project
### Windows 
 - requires MSVC 15 or above compiler.
 - requires a latest version of cmake.
 - build can be triggered by the following commands
 - 
        mkdir build
        cd build
        cmake .. -G "Visual Studio 16 2019" -A x64
        cmake --build . --config Release
        cd Release
        GNBClassifier.exe  

### Linux 
 - requires gnu gc++ 5.4 or above compiler.
 - requires a latest version of cmake.
 - build can be triggered by the following commands
 - 
        mkdir build
        cd build
        cmake .. && make
        ./GNBClassifier

## Introduction
This project deals with the theory of gaussian naive bayes classifier 
and it's implementation in C++. It uses an example data of a vehicle making some lane changes. 
The Gaussian NB classifier will predict the behavior of the vehicle on the highway given it's Frenet coordinates `s` and `d` and it's first order derivatives.  

## Theory
the Gaussian Naive Bayes classifier is an extension to the naive bayes classifier.
Abstractly, naïve Bayes is a conditional probability model: 
given a problem instance to be classified, represented by a vector `x = (x1, x2, x3 ... xn)` representing some n features (independent variables), 
it assigns to this instance probabilities  
  
![ds][formula1]  

for each of the `K` possible outcomes or classes `Ck`.  

The problem with the above formulation is that if the number of features n is large or if a feature takes on a large number of values, 
then basing such a model on probability tables is infeasible. The model must therefore be reformulated to make it more tractable.  

Using Bayes' theorem, the conditional probability can be decomposed as  

![ds][formula2]  

which is nothing but,

![ds][formula3]  

In practice, the numerator is only of interest as the denominator doesn't depend on `C` and the values `xi` are given which makes the denominator effectively constant.  

The numerator is equivalent to the joint probability model, which can be rewritten using the chain rule for repeated application of conditional probability

![ds][formula4]  

now, making a naiive assumption, all features `x` are mutually independent, conditional on the category `Ck` assuming,

![ds][formula5]  

Hence, the joint model can be espressed as:  

![ds][formula6]  

Thus, with the above independence assumptions, the conditional distribution over the class variable `C` is:  

![ds][formula7]  

### training the classifier

 For a feature `x` and label `C` with mean `μ` and standard deviation `σ`,
 the conditional probability can be computed using the formula

![ds][formula8]  

where, `v` would be used in the prediction step.  
`v` is the observed states of the vehicle which is used to find the conditional probability of `x` given `C` so that `C` given `x` can be found.

### prediction 

In this formula, the argmax is taken over all possible labels `Ck` and the product is taken over all features `Xi` with values `vi`.  

![ds][formula9]


## Code

`src/classifier.h` contains the class `GNB` which creates an instance of a gaussian naive bayes classifier object.  
   -  using the  `void train(...)` method the model is trained using the previously presented theory.  
   -  using the `string predict(...)` method, prediction can be done using the trained model.  
 **Note:** the member `possible_labels` would need to be extended/changed if data files have more/different labels.

## Data
In the image below the behaviors possible for on a 3 lane highway (with lanes of 4 meter width) is shown. 
The dots represent the d (y axis) and s (x axis) coordinates of vehicles as they either...

- change lanes left (shown in blue)
- keep lane (shown in black)
- or change lanes right (shown in red)

![ds][data1]  

the coordinate contains the following four features  
  - s
  - d
  - d(s)/dt
  - d(d)/dt 

the lane width is given as `4m`.

