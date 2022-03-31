%find zero test
%% Finding missclassifications
faults=guess.*known;%if a row in faults only has zeroes it was missclassified
for i=1:length(faults)
    mvect(i)=sum(faults(:,i));%zero in mvect indicates a missclassification at that index
end
mcind=find(~mvect);%Indexes of missclassified images
figure("Name",'Some mistakes made','NumberTitle','off')

%% Plotting some of the missclassifications
for i=1:15;
    X=zeros(28,28);
    subplot(3,5,i)
    index=mcind(i);
    X(:)=testv(index,:);
    x=fliplr(imrotate(X,270));
    image(x)
    guessed=find(guess(:,index))-1;
    title({sprintf('Guess %d',guessed),sprintf('Real: %d',testlab(index))});
end
