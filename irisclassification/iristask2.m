%% Prepping data and divding into classes.
% remove some features in the feature definition section
load("class_1");
load("class_2");
load("class_3");
%The number of points used for training or testing
ntrain=30;
ntest=20;

%definitions, After running plotfeatures.m and analysing results, we will
%remove feature 4, or the width of the petals
features=[1,3,4];
limit=0.6*5;
alpha=0.01;


nfeat=length(features);
nclass=3;

%seperating training and testing data
% To do task1a switch the indexes in these lines to fit the assigned 
% data-points
c1train=class_1(1:ntrain,features);
c1test=class_1(1+ntrain:ntrain+ntest,features);

c2train=class_2(1:ntrain,features);
c2test=class_2(1+ntrain:ntrain+ntest,features);

c3train=class_3(1:ntrain,features);
c3test=class_3(1+ntrain:ntrain+ntest,features);

%Collecting data and defining targets
training=[c1train;c2train;c3train];
test=[c1test;c2test;c3test];

T1=[1;0;0];
T2=[0;1;0];
T3=[0;0;1];

%% Setting up equation (22) from compendium(Classification section)
%W_MSE_k=@(gk,tk,xk) ((gk-tk).*gk.*(1-gk))*xk';
grad_W_MSE_k = @(gk, tk, xk) ( (gk - tk) .* gk .* (1 - gk) ) * xk';
sigmoid=@(x) (1./(1+exp(-x)));
%% Training the model
lim=limit;
t=[kron(ones(1,ntrain),T1),kron(ones(1,ntrain),T2),kron(ones(1,ntrain),T3)];
w=eye(nclass,nfeat+1);
condition=1;
iterations=0;
fprintf('Training commences \n')
tic;
while condition
    W_MSE=0;
    for k=1:nclass*ntrain
        xk=[training(k,:)'; 1];
        zk=w*xk;
        gk=sigmoid(zk);
        tk=t(:,k);
        W_MSE=W_MSE+grad_W_MSE_k(gk,tk,xk);
    end
    norm(W_MSE)
    condition=norm(W_MSE)>=lim;
    iterations=iterations+1;

    a=alpha;
    w=w-a*W_MSE;
end
timespent=toc;
fprintf('Training complete!\n Spent %3.6f s training \n',timespent)
fprintf('Used %i iterations to train!\n',iterations)

%% Testing the model trained above, and plotting results in a confusion matrix.
% Predictions available in test_pred, amount of correct classifications in
% ccn, where n is the class.
test_known=[kron(ones(1,ntest),T1),kron(ones(1,ntest),T2),kron(ones(1,ntest),T3)];
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

figure(1);
% Please note that the plotconfusion requires the deeplearning toolbox for
% matlab
plotconfusion(test_known,test_pred,'Test set, alpha=0.01');
titl = get(get(gca,'title'),'string');
title({titl, ['30 last points training,20 first points testing, alpha=0.01']});
xticklabels({'Setosa', 'Versicolour', 'Virginica'});
yticklabels({'Setosa', 'Versicolour', 'Virginica'});


figure(2);
plotconfusion(t,train_pred,'Training set');
titl = get(get(gca,'title'),'string');
title({titl, '30 last points training,20 first points testing, alpha=0.01'});
xticklabels({'Setosa', 'Versicolour', 'Virginica'});
yticklabels({'Setosa', 'Versicolour', 'Virginica'});

cc1=sum(test_pred(1,1:20));
cc2=sum(test_pred(2,21:40));
cc3=sum(test_pred(3,41:60));

