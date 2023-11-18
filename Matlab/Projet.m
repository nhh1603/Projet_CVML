
% Specify the paths to your training and validation data
DataDir = 'C:\Users\thaiv\OneDrive\Desktop\TIP\isic-2020-resized\train-resized\train-resized';
csvFile = 'C:\Users\thaiv\OneDrive\Desktop\TIP\isic-2020-resized\train-labels.csv';
dataTable = readtable(csvFile);
fileNames = fullfile(DataDir, dataTable.image_name + ".jpg");  % Add the file extension
target = categorical(dataTable.target);

imds = imageDatastore(fileNames, 'Labels', target);

% Count the number of images in each class
numClass0 = sum(imds.Labels == '0');
numClass1 = sum(imds.Labels == '1');

% Calculate the oversampling factor
oversampleFactor = ceil(numClass0 / numClass1);

% Find indices of images from the minority class
minorityIndices = find(imds.Labels == '1');

% Duplicate images from the minority class to balance the dataset
minorityImages = imds.Files(minorityIndices);
repeatedMinorityImages = repmat(minorityImages, oversampleFactor, 1);

% Remove the original minority samples from imds

    imds.Files(minorityIndices) = [];
    


% Create a new imageDatastore for oversampled data
filesOversampled = [imds.Files; repeatedMinorityImages];
labelsOversampled = [imds.Labels; repmat(categorical({'1'}), oversampleFactor * numel(minorityImages), 1)];
imdsOversampled = imageDatastore(filesOversampled, 'Labels', labelsOversampled);

% % Use augmentedImageDatastore for further data augmentation
% augmentedDatastore = augmentedImageDatastore([224 224], imdsOversampled, ...
%     'DataAugmentation', imageDataAugmenter, 'OutputSizeMode', 'randcrop', ...
%     'ColorPreprocessing', 'gray2rgb');

% Split the augmentedDatastore into training and validation sets
[trainingImds, validationImds] = splitEachLabel(imdsOversampled, 0.8, 'randomized');


% Define the CNN Architecture
net = resnet18;
%analyzeNetwork(net)
numClasses = numel(categories(trainingImds.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

% % Create an augmentedImageDatastore for training with data augmentation
% augmentedTrainingImds = augmentedImageDatastore([224,224,3],trainingImds, ...
%     'ColorPreprocessing', 'gray2rgb', ...
%     'DataAugmentation', imageDataAugmenter(...
%         'RandXReflection', true, ...
%         'RandYReflection', true, ...
%         'RandRotation', [-30, 30], ...
%         'RandScale', [0.8, 1.2], ...
%         'RandXTranslation', [-10, 10], ...
%         'RandYTranslation', [-10, 10], ...
%         'RandXShear', [-10, 10], ...
%         'RandYShear', [-10, 10], ...
%         'FillValue', 0));
    

% % Calculate class weights based on the number of samples in each class
% totalSamples = numel(trainingImds.Labels);
% numClass0 = sum(trainingImds.Labels == '0');  % Replace '0' with the actual label for the majority class
% numClass1 = sum(trainingImds.Labels == '1');  % Replace '1' with the actual label for the minority class






% Specify Training Options with GPU and parallel support
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 4, ...
    'MiniBatchSize', 32, ...
    'ValidationData', validationImds, ...
    'ValidationFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ... 
    'L2Regularization', 0.1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% Train the Network with augmented data using parallel processing
net = trainNetwork(trainingImds, lgraph, options);




% Perform binary classification predictions with probability scores
[ValPred, prob1] = classify(net, validationImds);
[TrainPred, prob2] = classify(net, trainingImds);

YValidation = validationImds.Labels;

YTrain=trainingImds.Labels;

incorrectIndices = find(ValPred ~= YValidation);
incorrectIndices1 = find(TrainPred ~= YTrain);

%Accuracy of the net work

    train0=(sum(TrainPred=='0')/sum(YTrain=='0'))
    train1=1/(sum(TrainPred=='1')/sum(YTrain=='1'))

    val0=(sum(ValPred=='0')/sum(YValidation=='0'))
    val1=1/(sum(ValPred=='1')/sum(YValidation=='1'))





% 
% % Extract probability of malignancy (assuming malignancy is the second class)
% probabilityOfMalignancy = probs(:, 2);
% 
% % Display the probability scores for the first few validation images
% disp('Probability of Malignancy:');
% disp(probabilityOfMalignancy(1:5));
accuracytrain = sum(TrainYpred == YTrain) / numel(YTrain)
accuracypred = sum(ValPred == YValidation) / numel(YValidation)
% % fprintf('Accuracy: %.4f\n', accuracy);

% Save the trained network
save Projet1.mat net lgraph;
