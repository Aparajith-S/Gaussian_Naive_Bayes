#include "classifier.h"
#include <cmath>
#include <string>
#include <vector>
#include<algorithm>
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif
using std::string;
using std::vector;

// Initializes GNB
GNB::GNB() : possible_labels{"left","keep","right"}{}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  //
  //Trains the classifier with N data points and labels.
  //@param data - array of N observations
  //  - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
  //  - Example : [[3.5, 0.1, 5.9, -0.02],
  //               [8.0, -0.3, 3.0, 2.2],
  //                ...
  //               ]
  //@param labels - array of N labels
  //  - Each label is one of "left", "keep", or "right".
  //
  //Implement the training function for your classifier.
    //compute prior probs
    m_priorProbs=findLabelCount(labels);
    for (const auto& label : possible_labels) 
    {
        m_priorProbs[label] /= labels.size();
    }
    //compute statistics for each label
    CalcStatistics(data,labels);

}

string GNB::predict(const vector<double> &sample) {
  //
  //Once trained, this method is called and expected to return 
  //  a predicted behavior for the given observation.
  //@param observation - a 4 tuple with s, d, s_dot, d_dot.
  //  - Example: [3.5, 0.1, 8.5, -0.2]
  //@output A label representing the best guess of the classifier. Can
  //  be one of "left", "keep" or "right".
    std::map<string, double> NBClassifier;
    for (const auto& label : possible_labels)
    {
        vector<double> probs;
        double product= m_priorProbs[label];
        for (int state = 0; state < sample.size(); state++)
        {
            double prob = gaussian(sample[state],
                m_statisticalData[label][state].mu,
                m_statisticalData[label][state].sigma);
            product *= prob;
        }
        NBClassifier[label]= product;
    }
    //find argmax and return
    return std::max_element(NBClassifier.begin(),
        NBClassifier.end(),
        [](const std::pair<string, double>& a, const std::pair<string, double>& b) {return (a.second < b.second); })->first;
}

std::map<string, double> GNB::findLabelCount(const vector<string>& labels) const
{
    //initialize the map
    std::map < string, double > priors;
    for (const auto & label : possible_labels)
    {
        priors.insert(std::make_pair(label, 0.0));
    }

    //begin counting!
    for (const auto & label : labels)
    {
        auto iter = priors.find(label);
        if (iter != priors.end())
        {
            iter->second++;
        }
        else
        {
            // throw is used here as the target is not intended to be used in a embedded uc.
            throw "FATAL: label mismatch between data read from file and possible labels. please check the file or the code"; 
        }
    }
    return priors;
}

void GNB::CalcStatistics(const vector<vector<double>>& data,
    const vector<string>& labels)
{
    std::map<string, vector<vector<double>>> DataTable;
    std::map < string, vector<vector <double>>> CondProbs;
    vector<stats> musigma;
    if (labels.size() != data.size())
    {
        throw "FATAL: training data and labels are of different lengths. check the file!";
    }
    for (int i = 0; i < data[0].size(); i++)
    {
        musigma.push_back(stats{ 0.0,0.0 });
    }
    for (const auto& label : possible_labels)
    {
        CondProbs.insert(std::make_pair(label, vector<vector <double>>()));
        DataTable.insert(std::make_pair(label, vector<vector <double>>()));
        m_statisticalData.insert(std::make_pair(label, vector<stats>{musigma}));
    }
    
    //find mean and stddev of data w.r.t. labels.
    // 1. extract data in a table-wise method and store in Statistical data. 
    //      each key contain --- vector([s ,d ,sdot ,ddot]) 
    // 2. compute the mean while inserting elements because it is faster. 
    // perform 1.
    for (int indx =0;indx<labels.size();indx++)
    {
        DataTable[labels[indx]].push_back(data[indx]);
        // perform 2. 
        for (int state = 0; state < data[indx].size(); state++)
        {
            m_statisticalData[labels[indx]][state].mu += data[indx][state];
        }   
    }
    // complete 2. 
    auto labelCount = findLabelCount(labels);
    for (const auto& label : possible_labels)
    {
        for (int state = 0; state < data[0].size(); state++)
        {
            m_statisticalData[label][state].mu /= labelCount[label];
        }
    }
    // 3.a. compute stddev for each label and each state
    for (const auto& label : possible_labels)
    {
        //compute expression = sum{pow((x_i - xBar),2)}   
        for (const auto& datum : DataTable[label])
        {
            for (int state = 0; state < datum.size(); state++) 
            {
                m_statisticalData[label][state].sigma+=pow((datum[state] - m_statisticalData[label][state].mu),2);
            }
        }
    }
    // 3.b. compute stddev = sqrt(expression/N)
    for (const auto& label : possible_labels)
    {
        for (int state = 0; state < data[0].size(); state++)
        {
            m_statisticalData[label][state].sigma /= labelCount[label];
            m_statisticalData[label][state].sigma = sqrt(m_statisticalData[label][state].sigma);
        }
    }
}

double GNB::gaussian(double x, double mu, double sigma) const
{
    double expr1 = 1.0 / sqrt(2 * M_PI);
    double expr2 = pow((x - mu) / sigma, 2)*-0.5;
    return (expr1 / sigma) * exp(expr2);
}

//EOF