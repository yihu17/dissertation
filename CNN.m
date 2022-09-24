clear all;
clc;

training = false;
if training == true
    datafolder = 'D:\Speech_Recognition_MatLab\strokes';
    %store all the audio files with audiDatastore
    data = audioDatastore(datafolder, 'IncludeSubfolders', true, 'FileExtensions', '.wav', 'LabelSource', 'foldernames');
    %store copy of data files
    datastoreCopy = copy(data);

    %%choosing words to recognise
    strokes = categorical(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]);
    %check if command of a subfolder 
    isCommand = ismember(data.Labels,strokes);

    %%split data into Training, Validation and Test
    [trainData, validationData, testData] = splitData(data, datafolder);

    %%Compute Speech Spectrograms
    segmentDuration = 1; %duration of each stroke clip in seconds
    frameDuration = 0.025; %duration of each frame for spectrogram
    hopDuration = 0.010; %time between each column of spectrogram
    numBands = 40; %number of frequency bands

    epsil = 1e-9; %offset/limit

    %Defining X and Y labels for neural network
    XTrain = speechSpectrograms(trainData, segmentDuration, frameDuration, hopDuration, numBands);
    %take log of spectrogram for smoother distribution
    XTrain = log10(XTrain + epsil);

    XValidation = speechSpectrograms(validationData, segmentDuration, frameDuration, hopDuration, numBands);
    XValidation = log10(XValidation + epsil);

    XTest = speechSpectrograms(testData, segmentDuration, frameDuration, hopDuration, numBands);
    XTest = log10(XTest + epsil);

    YTrain = trainData.Labels;
    YValidation = validationData.Labels;
    YTest = testData.Labels;

    %%Visualise sample data
    specMin = min(XTrain(:));
    specMax = max(XTrain(:));
    idx = randperm(size(XTrain,4),3);
    figure('Units','normalized', 'Position', [0.2 0.2 0.6 0.6]);
    %list 3 examples
    for i = 1:3
        %plot audio signal
        [x,fs] = audioread(trainData.Files{idx(i)});
        subplot(2,3,i)
        plot(x)
        axis tight
        title(string(trainData.Labels(idx(i))))
        %plot spectrogram 
        subplot(2,3,i+3)
        spect = XTrain(:,:,1,idx(i));
        pcolor(spect)
        caxis([specMin+2 specMax])
        shading flat
        %play the audio
        sound(x,fs)
        pause(2)
    end

    figure
    histogram(XTrain, 'EdgeColor', 'none', 'Normalization', 'pdf')
    axis tight
    ax = gca;
    ax.YScale = 'log';
    xlabel("Input Pixel Value")
    ylabel("Probability Density")

    %%Add background noise
    bkg = subset(datastoreCopy, datastoreCopy.Labels=="_background_noise_");
    numBkgClips = 2000;
    volumeRange = [1e-4,1];

    XBkg = backgroundSpectrograms(bkg, numBkgClips, volumeRange, segmentDuration, frameDuration, hopDuration, numBands);
    XBkg = log10(XBkg + epsil);

    numTrainBkg = floor(0.8*numBkgClips);
    numValidationBkg = floor(0.1*numBkgClips);
    numTestBkg = floor(0.1*numBkgClips);

    XTrain(:,:,:,end+1:end+numTrainBkg) = XBkg(:,:,:,1:numTrainBkg);
    XBkg(:,:,:,1:numTrainBkg) = [];
    YTrain(end+1:end+numTrainBkg) = "background";

    XValidation(:,:,:,end+1:end+numValidationBkg) = XBkg(:,:,:,1:numValidationBkg);
    XBkg(:,:,:,1:numValidationBkg) = [];
    YValidation(end+1:end+numValidationBkg) = "background";

    XTest(:,:,:,end+1:end+numTestBkg) = XBkg(:,:,:,1:numTestBkg);
    clear XBkg;
    YTest(end+1:end+numTestBkg) = "background";

    %removes unused categories from array
    YTrain = removecats(YTrain);
    YValidation = removecats(YValidation);
    YTest = removecats(YTest);

    figure('Units','normalized','Position', [0.2 0.2 0.5 0.5]);
    subplot(2,1,1)
    histogram(YTrain)
    title("Training Label Distribution")
    subplot(2,1,2)
    histogram(YValidation)
    title("Validation Label Distribution")

    %%Augment Data
    sz = size(XTrain);
    specSize = sz(1:2);
    imageSize = [specSize 1];
    augmenter = imageDataAugmenter('RandXTranslation', [-10 10], ...
        'RandXScale', [0.7 1.3], ...
        'FillValue', log10(epsil));
    augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',augmenter);

    %%Neural Network Architecture
    classWeights = 1./countcats(YTrain);
    classWeights = classWeights'/mean(classWeights);
    numClasses = numel(categories(YTrain));

    timePoolSize = ceil(imageSize(2)/27);
    dropoutProb = 0.5; 
    numFilters = 12;
    layers = [
        imageInputLayer(imageSize)
        %--------------------------------------------------
        convolution2dLayer(3,numFilters,'Padding','same')
        batchNormalizationLayer
        leakyReluLayer
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        %--------------------------------------------------
        convolution2dLayer(3,2*numFilters,'Padding','same')
        batchNormalizationLayer
        leakyReluLayer
        maxPooling2dLayer(3,'Stride',2,'Padding','same')
        %--------------------------------------------------
        convolution2dLayer(3,4*numFilters,'Padding','same')
        batchNormalizationLayer
        leakyReluLayer   
        maxPooling2dLayer([1 timePoolSize])
        %--------------------------------------------------
        dropoutLayer(dropoutProb)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        weightedClassificationLayer(classWeights)];

    %%Training neural network
    miniBatchSize = 128;
    validationFrequency = floor(numel(YTrain)/miniBatchSize);
    options = trainingOptions('adam',...
        'InitialLearnRate', 0.001,...
        'MaxEpochs', 10, ...
        'MiniBatchSize', miniBatchSize,...
        'Shuffle', 'every-epoch',...
        'ExecutionEnvironment', 'gpu',...
        'Plots', 'training-progress',...
        'Verbose', false, ...
        'ValidationData', {XValidation, YValidation},...
        'ValidationFrequency', validationFrequency,...
        'LearnRateSchedule', 'piecewise',...
        'LearnRateDropFactor', 0.1,...
        'LearnRateDropPeriod', 10);

    net = trainNetwork(augimdsTrain,layers,options);
    
    save cnn_mdl
    
    %conv1 = activations(trainedNet,XTest(:,:,1,1),'conv_1');
else
    load trainedNet
end

%%Evaluate Trained Network
YValPred = classify(trainedNet, XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet, XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Train error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

figure('Unit','normalized','Position',[0.2 0.2 0.5 0.5]);
matrix = confusionchart(YValidation, YValPred);
matrix.ColumnSummary = 'column-normalized';
matrix.RowSummary = 'row-normalized';
sortClasses(matrix, [strokes,"background"])

info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i = 1:100
    x = rand(imageSize);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment","gpu");
    time(i) = toc;
end
disp("Single-image prediction time on GPU: " + mean(time(11:end))*1000 + " ms")

%%Detecting strokes in real-time
fs = 16000;
classificationRate = 20;

audioIn = audioDeviceReader('SampleRate',fs,...
    'SamplesPerFrame', floor(fs/classificationRate),...
    'Device', 'Microphone (WO Mic Device)');

frameLength = floor(frameDuration*fs);
hopLength = floor(hopDuration*fs);
waveBuffer = zeros([fs,1]);

labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");
probBuffer = zeros([numel(labels),classificationRate/2]);

h = figure('Units','normalized','Position', [0.2 0.1 0.6 0.8]);

filterBank = designAuditoryFilterBank(fs,'FrequencyScale','bark',...
    'FFTLength', 512,...
    'NumBands', numBands,...
    'FrequencyRange', [50,7000]);

while ishandle(h)
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;
    
    [~,~,~,spec] = spectrogram(waveBuffer,hann(frameLength,'periodic'),frameLength - hopLength,512,'onesided');
    spec = filterBank * spec;
    spec = log10(spec + epsil);
    
    [YPredicted, probs] = classify(trainedNet,spec,'ExecutionEnvironment','gpu');
    YBuffer(1:end-1) = YBuffer(2:end);
    YBuffer(end) = YPredicted;
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = probs';
    
    subplot(2,1,1)
    plot(waveBuffer)
    axis tight
    ylim([-0.2, 0.2])
    
    subplot(2,1,2)
    pcolor(spec)
    caxis([specMin+2 specMax])
    shading flat
    
    [YMode, count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == YMode,:));
    probThreshold = 0.7;
    subplot(2,1,1);
    
    if YMode == "background" || count < countThreshold || maxProb < probThreshold
        title(" ")
    else 
        title(string(YMode),'FontSize',20)
    end
    
    drawnow
end
    