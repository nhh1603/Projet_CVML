%% Load the trained network
load Projet1_2.mat net lgraph;

%% Load the test images
DataDir = 'C:\Users\thaiv\OneDrive\Desktop\TIP\isic-2020-resized\test-resized\test-resized';
imds = imageDatastore(DataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'FileExtensions', '.JPG');

%% Initialize variables to store results
imageNames = cell(size(imds.Files, 1), 1);
probabilities = zeros(size(imds.Files, 1), 1);

%% Predictions for Test Images
for i = 1:size(imds.Files, 1)
    % Read and preprocess the image
    img = readimage(imds, i);
   
    
    % Make predictions
    [YPred, prob] = classify(net,img);
    
    % Extract probability of malignancy (assuming malignancy is the second class)
  
    probabilityOfMalignancy = round(prob(:, 2),1);
    
    
    % Extract the filename from the full path
    [~, fileName, ~] = fileparts(imds.Files{i});
    
    % Store results
    imageNames{i} = fileName;
    probabilities(i) = probabilityOfMalignancy;
    
    % Display probability scores and prediction
    fprintf('Image %d\n', i);
    fprintf('Probability of Malignancy: %.4f\n', probabilityOfMalignancy);
    fprintf('Predicted Class: %s\n\n', char(YPred));
end

%% Create a table with results
resultsTable = table(imageNames, probabilities, 'VariableNames', {'image_name', 'target'});

%% Write the table to a CSV file
writetable(resultsTable, 'results.csv');
