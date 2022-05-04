%% MNIST digit classification, data from Blackboard(using data_all.mat)
% Start by clearing everything else, and loading data
clear all
load("data_all.mat");
%full test/train data lengths
ntest=size(testv,1);
ntrain=size(trainv,1);
message='\nThis script will classify digits using data from assigned.\nClassification is done with NN-alogrithm\n';
fprintf(message)
%creating batches
testbatch=ntest/1;
trainbatch=ntrain/1;
fprintf('Testbatches of %i samples.\nTrainingbatches of %i samples. \n',testbatch,trainbatch)%Currently the code is set up for using the entire dataset

%% 1NN classification

guess=zeros(10,testbatch);%Like with the iris task, we will use a one-hot structure
tic;

% Purely cosmetic, We plot the digit being plotted and show a process bar since waiting 30 mins without any sign of progress is boring
name='Image to be classified.';
imfig=figure('Name',name);
ax1=axes(imfig);
X=zeros(28,28);
f = waitbar(0, 'Starting');

%classification:
for i=1:testbatch
    point=testv(i,:);
    distances=dist(trainv,point');%This and the next line is the actual classification, where we take the distance and find the minimum
    [dmin,minind]=min(distances);%Theory is given in the report
    prediction=trainlab(minind);
    guess(prediction+1,i)=1;
    X(:)=testv(i,:);
    x=fliplr(imrotate(X,270));%There are better("prettier") ways of displaying the image, but this is sufficient.
    image(ax1,x)
    waitbar(i/testbatch, f, sprintf(['\nClosing this will stop classification.\n Point %i was classified as %i'],i,prediction+1));
end
time=toc;
fprintf('Spent %2.2f minutes classifying using Nearest neighbour algorithm\n',time/60)%Since the programme spends several minutes, we divide by 60 to get appropriate units
close(f)
close(imfig)

% Preparing the true classes/labels
known=zeros(10,testbatch);
for i = 1:testbatch
    t=testlab(i);
    known(t+1,i) = 1;%One hot structure, as we use the plotconfusion command
end
% %% Plotting
%plottting confmat
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(known,guess);%Requires one-hot structured data
titl = get(get(gca,'title'),'string');
title({titl, '1-NN with no clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
