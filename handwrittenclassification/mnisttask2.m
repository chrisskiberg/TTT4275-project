%% MNIST digit classification using clustered data 
% Assumes that data is clustered and stored in correctly named variables
clear all
load data_all.mat
load full_set.mat
load clustered_data.mat

%% Setting up definitions
ntest=size(testv,1);
ntrain=size(clustereddata,1);%Here we switch the trainingdata with the clustered data, and thus change the classifier
                             %Not much else is different due to this intermediate step being used.
message=['\nThis script will classify digits using data from assigned.' ...
    '\nClassification is done with NN-alogrithm, on clustered data\n'];
fprintf(message)
%creating batches
testbatch=ntest/1;
trainbatch=ntrain/1;
fprintf(['Testbatches of %i samples.\nTrainingbatches of %i samples.\n' ...
    'Working with clustered data. \n'],testbatch,trainbatch)

%% 1NN classification (only on one testbatch)!

guess=zeros(10,testbatch);%One hot structure
tic;
fprintf('Starting classification\n');
% Purely cosmetic
name='Image to be classified.';
imfig=figure('Name',name);
ax1=axes(imfig);
X=zeros(28,28);
f = waitbar(0, 'Starting');

%classification:
for i=1:testbatch
    point=testv(i,:);
    distances=dist(clustereddata,point');%Classification is done here and line under. We use clustered data, as seen by the variable name.
    [dmin,minind]=min(distances);
    prediction=truecluster(minind);
    guess(prediction+1,i)=1;
    X(:)=testv(i,:);
    x=fliplr(imrotate(X,270));
    image(ax1,x)
    waitbar(i/testbatch, f, sprintf(['\nClosing this will stop classification.\n Point %i was classified as %i'],i,prediction));
end
time=toc;
fprintf('Spent %2.2f minutes classifying using Nearest neighbour algorithm\n',time/60)
close(f)
close(imfig)

% Preparing the true classes/labels
known=zeros(10,testbatch);
for i = 1:testbatch
    t=testlab(i);%Although we switched the training to clustered data, nothing happens to the testset, as this cant be clustered and still work as intended
                % (We would then not perform anything other than clustering on the test set and this is not desired.)
    known(t+1,i) = 1;%One hot
end
% %% Plotting
%plottting confmat
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(known,guess);
titl = get(get(gca,'title'),'string');
title({titl, '1-NN with M=64 clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
