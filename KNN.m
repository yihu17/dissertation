clear all;
clc;

redoDataset = false;
if redoDataset == true
    datafolder = 'D:\Speech_Recognition_MatLab\strokes';
    %store all the audio files with audiDatastore
    data = audioDatastore(datafolder, 'IncludeSubfolders', true, 'FileExtensions', '.wav', 'LabelSource', 'foldernames');
    %store copy of data files
    datastoreCopy = copy(data);

    %%choosing words to recognise
    strokes = categorical(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]);
    %check if command of a subfolder 
    isCommand = ismember(data.Labels,strokes);
    %otherwise classify them as unknown and add background noise
    isUnknown = ~ismember(data.Labels,[strokes,"_background_noise_"]);

    %only include a fraction of the unknown labels
    %balance out the between unknown and known labels
    includeFraction = 0.2;
    mask = rand(numel(data.Labels),1) < includeFraction;
    isUnknown = isUnknown & mask;
    data.Labels(isUnknown) = categorical("unknown");
    %reduce the datastore to only the files and labels in isCommand and isUnknown
    data = subset(data,isCommand|isUnknown);
    countEachLabel(data);

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

    %removes unused categories from array
    YTrain = removecats(YTrain);
    YValidation = removecats(YValidation);
    YTest = removecats(YTest);

    [m,~] = size(YTrain);
    X = zeros(m,3920);
    for index = 1:m
        X(index,:) = reshape(XTrain(:,:,1,index),1,[]);
    end

    mdl = fitcknn(X, YTrain, 'NumNeighbors',10);

    disp("testing model")
    [m,n] = size(YTest);
    test = zeros(m,3920);
    accuracy = 0;
    for index = 1:m
        test(index,:) = reshape(XTest(:,:,1,index),1,[]); 
        [label,score,cost] = predict(mdl,test(index,:));

        if label == YTest(index,1)
            accuracy = accuracy + 1;
        end
    end

    accuracy = (accuracy/m)*100;
    disp("accuracy of knn is " + accuracy + "%");
else
    load knn_mdl
end

%%Detecting strokes in real-time
fs = 16000;

rec = audiorecorder(fs, 16, 1, 1);

frameLength = floor(frameDuration*fs);
hopLength = floor(hopDuration*fs);

h = figure('Units','normalized','Position', [0.2 0.1 0.6 0.8]);

filterBank = designAuditoryFilterBank(fs,'FrequencyScale','bark',...
    'FFTLength', 512,...
    'NumBands', numBands,...
    'FrequencyRange', [50,7000]);

newX = zeros(1,3920);
dt = 1/fs;
asec = 16000;
tmpy = zeros(floor(asec),1);
letter = 'A';
Counter = 0;
cycle = 0;
while ishandle(h) || cycle < 100
    disp('Start Writing')
    recordblocking(rec, 2);
    disp('Stop Writing')
    
    y = getaudiodata(rec);
    
    t = 0:dt:(length(y)*dt)-dt;
    index = 1;
    asec = length(t)/3;
    
    while index <= length(y)
        if (y(floor(index)) >= 0.01) && (((index+asec)*dt) < 3) && floor(index-(asec/4)) > 300
            count = 1;
            tmp = floor(index-(asec/4));
            e = tmp + asec;
            while tmp <= e
                tmpy(count) = y(tmp);
                tmp = tmp + 1;
                count = count + 1; 
            end
            index = index + asec;
        end
        index = index + 1;
    end
    
    [~,~,~,spec] = spectrogram(tmpy,hann(frameLength,'periodic'),frameLength - hopLength,512,'onesided');
    spec = filterBank * spec;
    spec = log10(spec + epsil);
    
    subplot(2,1,1)
    plot(tmpy)
    axis tight
    ylim([-0.2, 0.2])
    
    subplot(2,1,2)
    pcolor(spec)
    caxis([specMin+2 specMax])
    shading flat
    
    newX = reshape(spec(:,:,1,1),1,[]);
    label = predict(mdl,newX);
    
    title(string(label),'FontSize',20)
    if label == letter 
        counter = counter + 1;
    end
    
    drawnow
    cycle = cycle + 1;
    pause(1)
end