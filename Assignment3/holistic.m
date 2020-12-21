%load and split the data
features = readmatrix("featuresFlowCapAnalysis.csv");
labels = readmatrix("labelsFlowCapAnalysis.csv");
train = features(1:length(labels),:);
test = features(length(labels)+1:end,:);

%set the seed so the plots are reproducable
rng(69)

%calculate the PCA
[pca_coeff,pca_score,~,~,pca_explained] = pca(train);
[pca_coeff_t,pca_score_t,~,~,pca_explained_t] = pca(test);

%embed the data
embedded = tsne(pca_score(:,1:3));
embedded_t = tsne(pca_score_t(:,1:3));

%plot the embedded data
figure(1);
scatter(embedded(labels == 1,1),embedded(labels == 1,2),25,'b','*')
hold on
scatter(embedded(labels == 2,1),embedded(labels == 2,2),25,'r','+')
title('tSNE on training data');
hold off

figure(2);
scatter(embedded_t(:,1),embedded_t(:,2),25,'black','.')
title('tSNE on test data');