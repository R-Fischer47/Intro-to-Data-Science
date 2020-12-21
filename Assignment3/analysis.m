%load and split the data
features = readmatrix("featuresFlowCapAnalysis.csv");
labels = readmatrix("labelsFlowCapAnalysis.csv");
train = features(1:length(labels),:);
test = features(length(labels)+1:end,:);

%set the seed so the plots are reproducable
rng(69)

%calculate the PCA
[pca_coeff,pca_score,~,~,pca_explained] = pca(train);
[pca_coeff_c,pca_score_c,~,~,pca_explained_c] = pca(features);
[pca_coeff_t,pca_score_t,~,~,pca_explained_t] = pca(test);

% plot the first two PCA of the training data
figure(1);
scatter(pca_score(labels==1,1),pca_score(labels==1,2),25,'b','*')
hold on
scatter(pca_score(labels==2,1),pca_score(labels==2,2),25,'r','+')
title('pca score training data');
hold off

% plot the percentage of variance for the training data
figure(2);
plot(pca_explained_c(1:40))
title('percentage of variance complete data')
xlabel('principal componants')
ylabel('%')

% plot the percentage of variance for the complete data
figure(3);
plot(pca_explained(1:40))
title('percentage of variance train data')
xlabel('principal componants')
ylabel('%')

%find the most important features using one way anova
Fstat = zeros(1,size(train,2));
most_valuable = 1:size(train,2);

for i=1:size(train,2)
    [~,tbl] = anova1(train(:,i),labels,'off');
    Fstat(i) = tbl{2,5};
end

[~,srtidx] = sort(Fstat,'descend');
Fstat = Fstat(srtidx);
most_valuable = most_valuable(srtidx);

%plot the most important features with a box-plot
for i= 1:4
   feature = most_valuable(i);
   anova1(train(:,feature),labels);
   title(['feature: ' string(feature) ' F: ' string(Fstat(i))]);
end

%plot the least important features with a box-plot
for i= length(most_valuable)-4:length(most_valuable)
    feature = most_valuable(i);
    anova1(train(:,feature),labels);
    title(['feature: ' string(feature) ' F: ' string(Fstat(i))]);
end



