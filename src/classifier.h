#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include<map>
using std::string;
using std::vector;

class GNB {
public:

    //Constructor
    GNB();

    //Destructor
    virtual ~GNB();

    /// @brief Train classifier
    //  @param 2D vector data
    //
    void train(const vector<vector<double>>& data,
        const vector<string>& labels);

    /**
     * Predict with trained classifier
     */
    string predict(const vector<double>& sample);
    
    vector<string> possible_labels;
private:
    std::map<string, double> findLabelCount(const vector<string>&labels)const;
    void CalcStatistics(const vector<vector<double>>& data,
        const vector<string>& labels);
    double gaussian(double x, double mu, double sigma)const;
    struct stats 
    {
        double mu;
        double sigma;
    };
    std::map<string, double> m_priorProbs;
    std::map < string, vector<stats>> m_statisticalData;
};

#endif  // CLASSIFIER_H