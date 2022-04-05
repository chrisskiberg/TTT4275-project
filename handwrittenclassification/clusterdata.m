%% Prepare workspace, initialise variables
% Only assume the correct variables are able to be loaded correctly
clear all
load data_all.mat
load full_set.mat
sorteddata=cell(10,1);

M=64;%From task
clusters=cell(10,1);


%% Sort data into correct labels
for i=0:9
    sorteddata{i+1}=trainv(trainlab==i,:); %Sort data into correct classes
end

%% Create the clusters
fprintf('\nClustering, please hold. Will spend ~30 seconds \n')
tic;
for i=1:10
    [temp,C_i]=kmeans(sorteddata{i},M);
    clusters{i}=C_i;
end
timespent=toc;
clear temp;
fprintf('Finished clustering.\nSpent %2.2f seconds clustering\n',timespent)

clustereddata=cell2mat(clusters);

%% We now need to create the knowns for the clusters.
% Since M=64, the first 64 are 0, second 64 is 1 etc.
truecluster=zeros(10*M,1);
for i=0:9
    truecluster((i*M)+1:(i+1)*M)=i*ones(M,1);
end


