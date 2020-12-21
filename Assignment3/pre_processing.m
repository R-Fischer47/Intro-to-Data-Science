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

%find the most important features
Fstat = zeros(1,size(train,2));
most_valuable = 1:size(train,2);

for i=1:size(train,2)
    [~,tbl] = anova1(train(:,i),labels,'off');
    Fstat(i) = tbl{2,5};
end

[~,srtidx] = sort(Fstat,'descend');
Fstat = Fstat(srtidx);
most_valuable = most_valuable(srtidx);

%plot the original distribution
embedded = tsne(train);
PlotPreProcessing(embedded, labels, 'original',1);

%apply feature selection
%we select only the 40 most important variables
embedded = tsne(train(:,most_valuable(1:40)));
PlotPreProcessing(embedded, labels, 'feature selection',2)

%apply z-score
embedded = tsne(zscore(train));
PlotPreProcessing(embedded, labels, 'z-score',3)

%PCA with leading 40 components
embedded = tsne(pca_score(:,1:40));
PlotPreProcessing(embedded, labels, 'PCA with leading 40 eigenvectors',4)

function PlotPreProcessing(embedded, labels, tit, fig)
    figure(fig)
    scatter(embedded(labels == 1,1),embedded(labels == 1,2),25,'b','*')
    hold on
    scatter(embedded(labels == 2,1),embedded(labels == 2,2),25,'r','+')
    title(tit);
    hold off
end

