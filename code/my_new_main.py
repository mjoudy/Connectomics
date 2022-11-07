#I made new main in order to clear some parts which because of lots of errors didn't work.
#I think it was some syntax srrors and some changes in version of some dependencies which no longer get supports. 
# The main part cleared is functions needed to write results on a .csv file

import sys
import time
import os
import numpy
import scipy
import csv
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.sparse import *
from sklearn import metrics
from datetime import date
from datetime import datetime
from numpy import *
from scipy import sparse
import scipy.io as sio
from io import StringIO

from plotROC import plotROC
from reshapeScoresMatrix import reshapeScores
from reshapeNetwork import reshapeNetwork
from iter_loadtxt import iter_loadtxt
from writeNetworkScoresInCSV import writeNetworkScoresInCSV
from randomScoreCode import randomScore
from computeCrossCorrelation import computeCrossCorrelation
from readNetworkScoresCode import readNetworkScores
from computePearsonsCorrelation import computePearsonsCorrelation
from computeIGCI import computeIGCI
from computeMI import computeMI

filesep = '/'

tic = time.clock();

funcdict = {

  'randomScore': randomScore,

  'crossCorrelation':computeCrossCorrelation,

  'pearsonsCorrelation':computePearsonsCorrelation,

  'information-geometry-causal-inference':computeIGCI,

  'mutualinformation': computeMI

}

def main():

    default_path = '/home/joudy/Documents/Codes/connectomics/Connectomics/code/'
    dataDirectory = '/home/joudy/Documents/Codes/connectomics/Connectomics/data/'
    submissionDirectory = '/home/joudy/Documents/Codes/connectomics/Connectomics/results'

    file_count = 1
    networkIdNames = []

    networkIdNames.append('mockvalid')
    networkIdNames.append('mocktest')

    scoringMethods = [];
    scoringMethods.append('pearsonsCorrelation');

    modelName = 'sample_model'
    concatenateScores = 0;

    if not os.path.exists(submissionDirectory):
        os.makedirs(submissionDirectory)

    the_date = datetime.now()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    extension='.txt';
    logfile = submissionDirectory + filesep + 'logfile.txt'
    flog=open(logfile, 'a');
    print('==========================================================\n')
    print('\n ChaLearn connectomics challenge, sample code version '+sys.version+'\n')
    print(' Date and time started: ' + the_date.strftime("%d/%m/%Y %H:%M:%S")+'\n')
    print(' Saving AUC results in ' + logfile+'\n')
    print('==========================================================\n\n')

    start = time.time()

    metNum=len(scoringMethods);
    netNum=len(networkIdNames);
    scores = numpy.empty((netNum,metNum))

    for j in range(0,metNum):
        scoringMethod = scoringMethods[j];
        target = submissionDirectory + filesep + scoringMethod + '_' +"_".join(str(x) for x in networkIdNames )+timestr+'_kaggle_ready.csv';
        tf = open(target, 'a')
        scoreFile = submissionDirectory + filesep + scoringMethod + '_' +"_".join(str(x) for x in networkIdNames )+timestr+'.csv'; 

        for i in range(0,netNum):
            networkId = networkIdNames[i];
            print('***'+ scoringMethod  +' on '+networkId+' ***\n\n' );

            fluorescenceFile = dataDirectory + filesep+ 'fluorescence_'+networkId+extension;
            F = iter_loadtxt(fluorescenceFile)

            tic = time.clock()
            print('Computing scores with ' + scoringMethod + '\n')

            if scoringMethod == 'trainedPredictor':
                arg =  modelDirectory + filesep + modelName + '.mat'
            else:
                arg = 'false'

            scores = funcdict[scoringMethod](F, arg)
            toc = time.clock()


            networkFile = dataDirectory + filesep + 'network_' + networkId + extension

            if os.path.exists(networkFile):
                print ('Computing ROC with using network ' + networkFile + '\n')
                network = readNetworkScores(networkFile);

                if scoringMethod == 'randomScore':
                    scores_dense =  scores.toarray()
                else:
                    scores_dense =  scores

                pred = reshapeScores(scores_dense)
                true = reshapeNetwork(network)

                fpr, tpr, thresholds = metrics.roc_curve(true,pred)
                print('\n==> AUC = '+ str(metrics.auc(fpr,tpr))+'\n');


                resuFile = scoringMethod+'_'+networkId+'_'+timestr+'.png'
                fullpath = os.path.join(submissionDirectory , resuFile)
                plotROC(fpr,tpr,fullpath)

                flog.write(the_date.strftime("%d/%m/%Y %H:%M:%S")+'\t'+scoringMethod+'\t'+networkId+'\t'+'%.4f\n' % ((metrics.auc(fpr, tpr))))

        end = time.time()
        print (end - start)
        print (the_date.strftime("%d/%m/%Y %H:%M:%S")+' Challenge solved.' +'\n')
        toc = time.clock();
        flog.close()
        tf.close()

if __name__=="__main__":
    main()
    
