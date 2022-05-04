%% Preparing correctly prepared images
% assume this is run after "findshowmistakes.m"

%This is pretty much the same script as findshowmistakes, but with a different method of selecting images
%We just pick a "random" image, and then check if it is missclassified. if it is wejust pick a new one.


%find "random" correctly identified images
%rng(5679003) random number chosen as seed, just if we want to controll
%reproducability
cind=randi(length(guess),1,15);
for i=1:15;
    if ismember(cind(i),mcind)
        cind(i)=cind(i)+1;
    end
end
%Note that since this is not part of the actual task, and the set is so 
% large(>1000), the chance of a number appearing twice is so low that 
% i have not taken this into consideration. run the code twice if it
% happens
%% Plotting found images
figure("Name",'Correctly classified images','NumberTitle','off')
for i=1:10;
    X=zeros(28,28);
    subplot(2,5,i)
    index=cind(i);
    X(:)=testv(index,:);
    x=fliplr(imrotate(X,270));
    image(x)
    guessed=find(guess(:,index))-1;
    title({sprintf('Guess: %d, real: %d',guessed,testlab(index)),sprintf('Image: %d',index)});
end
