% Genetic algorithm for the selection of the best subset of features
% Course: Introduction to Data Science
% Author: George Azzopardi
% Date:   October 2019

function bestchromosome = myGeneticAlgorithm(features,labels)
% features = A matrix of independent variables
% labels = A vector that contains the labels for each rows in matrix features


nchroms       = 100; % number of chromosomes
nepochs       = 10;  % number of epochs
nparentsratio = 0.2; % portion of elite list
mutateprob    = 0.1; % probability to mutate a bit in a chromosome

% Create figure that shows the progress of the genetic algorithm
figure;hold on;
title('Feature Selection with Genetic Algorithm');
colorlist = jet(nepochs);

% Convert labels, which can be in string format, to numeric.
[lbls,h] = grp2idx(labels);

% Iterate through all epochs
for epoch = 1:nepochs
    fprintf('epoch %d of %d\n',epoch,nepochs);
    if epoch == 1
        % generate the intial popultion of chromosome with randomly
        % assigned bits
        pop = generateInitialPopulation(nchroms,size(features,2));        
    else
        % generate a new population by creating offspring from the best
        % performing chromosome (or parents)
        pop = getnewpopulation(pop,score,nparentsratio,mutateprob);
    end    
    pop = logical(pop);
    
    % Compute the fitness score for each chromosome
    score = zeros(1,nchroms);
    for i = 1:nchroms
        score(i) = getScore(pop(i,:),features,lbls);    
    end    
    
    % Plot the scores to visualize the progress
    plot(sort(score,'descend'),'color',colorlist(epoch,:));
    xlabel('Chromosome');
    ylabel('Fitness Score');
    legendList{epoch} = sprintf('Epoch %d',epoch);
    legend(legendList);
    drawnow;
end

% Return the chromosome with the maximum fitness score
[~,mxind] = max(score);
bestchromosome = pop(mxind,:);

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
% Generate a new population by first selecting the best performing
% chromosomes from the given pop matix, and subsequently generate new offspring chromosomes from randomly
% selected pairs of parent chromosomes.

% Step 1. Write code to select the top performing chromosomes. Use nparentsratio to
% calculate how many parents you need. If pop has 100 rows and
% nparentsration is 0.2, then you have to select the top performing 20
% chromosomes
[~,ind] = sort(score,'descend');
nparents = nparentsratio * size(pop,1);

newpop = zeros(size(pop));
newpop(1:nparents,:) = pop(ind(1:nparents),:);

topparents = pop(ind(1:nparents),:);

% Step 2. Iterate until a new population of the same size is generated. Using the above
% example, you need to iterate 80 times. In each iteration create a new
% offspring chromosome from two randomly selected parent chromosomes. Use
% the function getOffSpring to generate a new offspring.

for j = nparents+1:size(pop,1)    
    randparents = randperm(nparents);    
    newpop(j,:) = getOffSpring(topparents(randparents(1),:),topparents(randparents(2),:),mutateprob);    
end

function offspring = getOffSpring(parent1,parent2,mutateprob)
% Generate an offspring from parent1 and parent2 and mutate the bits by
% using the probability mutateprob.
offspring = pointCrossover(parent1,parent2);
offspring = mutate(offspring, mutateprob);


function offspring = pointCrossover(parent1,parent2)
% Splits the parent chromosomes in half and assigns the left half of parent
% 1 to the left part of the offspring and the right part of parent2 to the
% right part of the offspring chromosome.
splitLength = fix(length(parent1)/2);
offspring = [parent1(1:splitLength) parent2(splitLength+1:length(parent1))];

function offspring = mutate(offspring, mutateprob)
% Iterates through the offspring chromosome and checks for every bit
% whether the random number is below the mutation probability. If it is, it
% flips the bit. 
for i = length(offspring)
    if mutateprob < rand()
        offspring(i) = ~offspring(i);
    end 
end

% Accidentally implemented the PMX crossover algorithm because I thought we
% were dealing with an ordered list. The selection of the randomly selected
% cutoff may still be buggy (might select whole chromosome of parent1).

% function offspring = pmxCrossover(parent1,parent2)
% lengthC     = length(parent1);
% cutoff      = sort(randsample(lengthC,2));
% lowerLim    = cutoff(1);
% upperLim    = cutoff(2);
% offspring   = NaN(lengthC);
% offspring   = parent1(:,lowerLim:upperLim);
% difference  = find(~ismember(parent1(lowerLim:upperLim),parent2(lowerLim:upperLim)));
% for i = difference
%     j = i;
%     while ~isnan(offspring(j))
%         value   = parent1(j);
%         j       = find(value,parent2);
%     end 
%     offspring(j) = parent2(j);
% end 
% for k = find(isnan(offspring))
%     offspring(k) = parent(k);
% end
% 
% function offspring = mutateOrderedList(offspring, mutateprob)
% if mutateprob < rand()
%     switchIndex = randsample(length(offspring),2);
%     a = offspring(switchIndex(1));
%     offspring(switchIndex(1)) = offspring(switchIndex(2));
%     offspring(switchIndex(2)) = a;
% end 

function score = getScore(chromosome,train_feats,labels)
% Compute the fitness score using 2-fold cross validation and KNN
% classifier
cv = cvpartition(labels,'Kfold',2);
for i = 1:cv.NumTestSets        
    knn = fitcknn(train_feats(cv.training(i),chromosome),labels(cv.training(i)));
    c = predict(knn,train_feats(cv.test(i),chromosome));
    acc(i) = sum(c == labels(cv.test(i)))/numel(c);
end
meanacc = mean(acc);
sumOfZeros = sum(chromosome(:)==0);
score = (power(10,4)*meanacc) + (0.4*sumOfZeros);

function pop = generateInitialPopulation(n,ndim)
% Generates an initial population based on n (how many members) and ndim
% (how long is each chromosome) 
for i = 1:n 
    pop(i,:) = randi([0, 1], [1, ndim]);
end
