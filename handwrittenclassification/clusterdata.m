%% Sort data into appropriate groups according to class
sorteddata=cell(10,1);
for n=0:9
    sorteddata{n+1}=trainv(trainlab==n,:); %Sort data into correct classes
end

%% Create the clusters
M=64;%From task
clusters=cell(10,1);

fprintf('\nClustering, please hold. Will spend ~30 seconds \n')
tic;
for i=1:10
    [ind,C_i]=kmeans(sorteddata{i},M);
    clusters{i}=C_i;
end
timespent=toc;
fprintf('Finished clustering.\nSpent %2.2f seconds clustering\n',timespent)