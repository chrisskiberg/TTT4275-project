%find zero test
%% Finding missclassifications
faults=guess.*known;%if a row in faults only has zeroes it was missclassified
for i=1:length(faults)
    mvect(i)=sum(faults(:,i));%zero in mvect indicates a missclassification at that index
end
mcind=find(~mvect);%Indexes of missclassified images, as find(~) gives indexes of 0-elements
figure("Name",'Some mistakes made','NumberTitle','off')

%% Plotting some of the missclassifications
for i=1:10;
    X=zeros(28,28);
    subplot(2,5,i)%For proper placement
    index=mcind(i);
    X(:)=testv(index,:);%Select a single image that was misclassified
    x=fliplr(imrotate(X,270));%We rotate it so that we can look at it normally
    image(x)%Plot image
    guessed=find(guess(:,index))-1;%for the name/title
    title({sprintf('Guess %d',guessed),sprintf('Real: %d',testlab(index))});
end
