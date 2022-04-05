%% MNIST digit classification using clustered data 
% Assumes that data is clustered and stored in correctly named variables
clear all
load data_all.mat
load full_set.mat
load clustered_data.mat

%% Setting up definitions
k=7;
ntest=size(testv,1);
ntrain=size(clustereddata,1);

message=['\nThis script will classify digits usign data from assigned.' ...
    '\nClassification is done with %i NN-alogrithm, on clustered data.\n'];
fprintf(message,k)
%creating batches
testbatch=ntest/1;
trainbatch=ntrain/1;
fprintf(['Testbatches of %i samples.\nTrainingbatches of %i samples.\n\n'],testbatch,trainbatch)

%% 1NN classification (only on one testbatch)!
% This section also plots the confmat, as just classification is
% meaningless unless we can display the results properly

guess=zeros(10,testbatch);
tic;
fprintf('Starting classification\n');

%classification:
for i=1:testbatch
    point=testv(i,:);
    distances=dist(clustereddata,point');

    [temp,minind]=sort(distances);%"Sort" places lowest values lowest
    ksmallest=minind(1:k);%Take the indexes from the lowest distances

    setedge=[-0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5];
    [N,edges]=histcounts(truecluster(ksmallest),setedge);
    [val,ind]=max(N);

    %prediction=truecluster(minind);
    prediction=ind-1;
    guess(prediction+1,i)=1;
end
time=toc;
fprintf('Spent %2.2f minutes classifying using Nearest neighbour algorithm\n',time/60)


% Preparing the true classes/labels
known=zeros(10,testbatch);
for i = 1:testbatch
    t=testlab(i);
    known(t+1,i) = 1;
end
% %% Plotting
%plottting confmat
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(known,guess);
titl = get(get(gca,'title'),'string');
title({titl, '7-NN with M=64 clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
