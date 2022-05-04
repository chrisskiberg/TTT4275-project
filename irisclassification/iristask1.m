%Data from BB, Iris(flower) classification script
%% preparing data, and seperating into training and testing subsets
load("class_1");
load("class_2");
load("class_3");
%The number of points used for training or testing
ntrain=30;
ntest=20;

%definitions
nfeat=4; % Using full feature set
nclass=3;% 3 classes since there are three flowers

limit=0.6;
alpha=0.01;

%seperating training and testing data
% To do task1a switch the indexes in these lines to fit the assigned 
% data-points
c1train=class_1(1:ntrain,:);
c1test=class_1(ntrain+1:ntest+ntrain,:);

c2train=class_2(1:ntrain,:);
c2test=class_2(ntrain+1:ntest+ntrain,:);

c3train=class_3(1:ntrain,:);
c3test=class_3(ntrain+1:ntest+ntrain,:);
% 
% c1train=class_1(ntest+1:ntest+ntrain,:);
% c1test=class_1(1:ntest,:);
% 
% c2train=class_2(ntest+1:ntest+ntrain,:);
% c2test=class_2(1:ntest,:);
% 
% c3train=class_3(ntest+1:ntest+ntrain,:);
% c3test=class_3(1:ntest,:);

%Collecting data and defining targets
training=[c1train;c2train;c3train];%All training data for all classes is collected here, first class 1 then class2 etc
test=[c1test;c2test;c3test];%Mirrors set above, but with Test data

T1=[1;0;0];%These arefor setting up the labels later. 
T2=[0;1;0];
T3=[0;0;1];

%% Setting up equation (22) from compendium(Classification section), and the sigmoid to be used later
grad_W_MSE_k = @(gk, tk, xk) ( (gk - tk) .* gk .* (1 - gk) ) * xk';
sigmoid=@(x) (1./(1+exp(-x)));
%% Training the model

t=[kron(ones(1,ntrain),T1),kron(ones(1,ntrain),T2),kron(ones(1,ntrain),T3)];%Here we set up our labels, in the one-hot method.
w=eye(nclass,nfeat+1);
c=1;

fprintf('Training commences \n')
tic;
while c
    W_MSE=0;
    for k=1:nclass*ntrain%Here we realise the theory given in the report, approx section 2
        xk=[training(k,:)'; 1];
        zk=w*xk;
        gk=sigmoid(zk);
        tk=t(:,k);
        W_MSE=W_MSE+grad_W_MSE_k(gk,tk,xk);
    end
    c=norm(W_MSE)>=limit;
    %Update the weights after the entire test set is run through. This could be done at more regular steps, but this was not inlcuded in the task.
    %This is known as batches in general ML.
    w=w-alpha*W_MSE;
end
timespent=toc;
fprintf('Training complete!\n Spent %3.6f s training \n',timespent)

%% Testing the model trained above, and plotting results in a confusion matrix.
% Predictions available in test_pred, amount of correct classifications in
% ccn, where n is the class.
test_known=[kron(ones(1,ntest),T1),kron(ones(1,ntest),T2),kron(ones(1,ntest),T3)];%Setting up testlabels and again using one-hot.
test_pred=zeros(size(test_known));

for i=1:length(test)
    x=[test(i,:)'; 1];
    g=sigmoid(w*x);
    [f,j] = max(g);
    test_pred(j,i)=1;
end

train_pred=zeros(size(t));
for i=1:length(training)
    x=[training(i,:)'; 1];
    g=sigmoid(w*x);
    [f,j] = max(g);
    train_pred(j,i)=1;
end    

figure(1);%Prediction for the test set
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(test_known,test_pred,'Test set, alpha=0.01');%As mentioned, this function requires the data to be in a one-hot structure, which it is
titl = get(get(gca,'title'),'string');
title({titl, ['30 last points training,20 first points testing, alpha=0.01']});
xticklabels({'Setosa', 'Versicolour', 'Virginica'});
yticklabels({'Setosa', 'Versicolour', 'Virginica'});


figure(2);%Prediction for the training set
plotconfusion(t,train_pred,'Training set');
titl = get(get(gca,'title'),'string');
title({titl, '30 last points training,20 first points testing, alpha=0.01'});
xticklabels({'Setosa', 'Versicolour', 'Virginica'});
yticklabels({'Setosa', 'Versicolour', 'Virginica'});

cc1=sum(test_pred(1,1:20));%Here the sum of correctly classified points will be.
cc2=sum(test_pred(2,21:40));
cc3=sum(test_pred(3,41:60));
