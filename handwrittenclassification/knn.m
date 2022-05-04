%% MNIST digit classification using clustered data 
% Assumes that data is clustered and stored in correctly named variables
clear all
load data_all.mat
load full_set.mat
load clustered_data.mat

%% Setting up definitions
k=7;%Define how many neighbours to consider
ntest=size(testv,1);
ntrain=size(clustereddata,1);%We are performing the classification with clustered data

message=['\nThis script will classify digits usign data from assigned.' ...
    '\nClassification is done with %i NN-alogrithm, on clustered data.\n'];
fprintf(message,k)
%creating batches
testbatch=ntest/1;
trainbatch=ntrain/1;
fprintf(['Testbatches of %i samples.\nTrainingbatches of %i samples.\n\n'],testbatch,trainbatch)

%% 1NN classification


guess=zeros(10,testbatch);
tic;
fprintf('Starting classification\n');

%classification:
for i=1:testbatch
    point=testv(i,:);
    distances=dist(clustereddata,point');

    [temp,minind]=sort(distances);%"Sort" places lowest values at low indexes
    ksmallest=minind(1:k);%Take the indexes from the lowest distances
                         % Since we are to use 7 neighbours, we take the seven lowest indexes and use them for our prediction
                         % this is where the 7-NN part is visible. other than this(and l. 38->40) its not large changes

    setedge=[-0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5];   %There is most likely a prettier version of doing this, but this works.
    [N,edges]=histcounts(truecluster(ksmallest),setedge);% This function counts the values in the ksmallest vector
    [val,ind]=max(N);%here we take the class that is most frequent as defined above. This is our prediction

    prediction=ind-1;
    guess(prediction+1,i)=1;
end
time=toc;
fprintf('Spent %2.2f minutes classifying using Nearest neighbour algorithm\n',time/60)%being scarred from the ~30 mins spent by 1nn unclustered we still use minutes
                                                                                      % but the script use seconds instead.


% Preparing the true classes/labels
known=zeros(10,testbatch);
for i = 1:testbatch
    t=testlab(i);
    known(t+1,i) = 1;%One hot structure
end
% %% Plotting
%plottting confmat
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(known,guess);%requires one hot structure
titl = get(get(gca,'title'),'string');
title({titl, '7-NN with M=64 clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
