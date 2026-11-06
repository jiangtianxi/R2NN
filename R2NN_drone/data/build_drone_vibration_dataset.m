%% Create the three-class drone vibration dataset
% This is a script, not a function. The original 10240 Hz signals are split
% directly without downsampling. Class order is fixed explicitly:
%   First class  <- 4full_1500_1545.txt
%   Second class <- 4full_1500_1438.txt
%   Third class  <- 4full_1300_1335.txt

clear;
clc;

%% Configuration
samplingFrequency = 10240;
segmentDurationSeconds = 1;
samplesPerSegment = samplingFrequency * segmentDurationSeconds;
trainingSamplesPerClass = 240;
testSamplesPerClass = 60;
samplesPerClass = trainingSamplesPerClass + testSamplesPerClass;
randomSeed = 20260811;

classNames = {'第一类', '第二类', '第三类'};
classFileNames = { ...
    '4full_1500_1545.txt', ...
    '4full_1500_1438.txt', ...
    '4full_1300_1335.txt'};

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    dataDirectory = pwd;
else
    dataDirectory = fileparts(scriptPath);
end
rawDirectory = fullfile(dataDirectory, 'drone_vib_raw_data');
trainingDirectory = fullfile(dataDirectory, 'train');
testDirectory = fullfile(dataDirectory, 'test');

if ~isfolder(rawDirectory)
    error('Raw-data directory does not exist: %s', rawDirectory);
end
if ~isfolder(trainingDirectory)
    mkdir(trainingDirectory);
end
if ~isfolder(testDirectory)
    mkdir(testDirectory);
end

for classId = 1:numel(classFileNames)
    rawPath = fullfile(rawDirectory, classFileNames{classId});
    if ~isfile(rawPath)
        error('Missing raw-data file for %s: %s', ...
            classNames{classId}, rawPath);
    end
end

oldRandomState = rng;
restoreRandomState = onCleanup(@() rng(oldRandomState)); %#ok<NASGU>
rng(randomSeed, 'twister');

classInfo = repmat(struct( ...
    'classId', [], ...
    'className', '', ...
    'sourceFile', '', ...
    'rawSampleCount', [], ...
    'trainingSampleCount', [], ...
    'testSampleCount', [], ...
    'guardSampleCount', [], ...
    'guardDurationSeconds', [], ...
    'trainingStartIndices', [], ...
    'testStartIndices', []), 1, numel(classFileNames));

fprintf('Creating dataset directly from the original %d Hz signals.\n', ...
    samplingFrequency);
fprintf('Each class: %d training + %d independent-test samples.\n\n', ...
    trainingSamplesPerClass, testSamplesPerClass);

%% Read, split and save every class
for classId = 1:numel(classFileNames)
    sourceFile = classFileNames{classId};
    rawPath = fullfile(rawDirectory, sourceFile);

    fileId = fopen(rawPath, 'rt');
    if fileId == -1
        error('Cannot open raw-data file: %s', rawPath);
    end
    closeFile = onCleanup(@() fclose(fileId));

    header = cell(7, 1);
    for lineIndex = 1:7
        header{lineIndex} = fgetl(fileId);
        if ~ischar(header{lineIndex})
            error('%s ended before its 7-line header was read.', rawPath);
        end
    end
    if ~strcmpi(strtrim(header{7}), 'data:')
        error('%s has an unexpected header; line 7 must be "data:".', ...
            rawPath);
    end
    values = textscan(fileId, '%f');
    rawSignal = values{1};
    clear closeFile

    if isempty(rawSignal)
        error('No numeric signal data found in %s.', rawPath);
    end
    if any(~isfinite(rawSignal))
        error('%s contains NaN or Inf values.', sourceFile);
    end

    rawSampleCount = numel(rawSignal);
    requiredSampleCount = samplesPerClass * samplesPerSegment;
    if rawSampleCount < requiredSampleCount
        error(['%s has %d points, but %d are required for %d ' ...
            'non-overlapping one-second samples.'], ...
            sourceFile, rawSampleCount, requiredSampleCount, samplesPerClass);
    end

    % Use the beginning for training and the tail for independent testing.
    trainingStartIndices = ...
        1 + (0:(trainingSamplesPerClass - 1)) * samplesPerSegment;
    trainingEndIndex = trainingSamplesPerClass * samplesPerSegment;
    testFirstIndex = ...
        rawSampleCount - testSamplesPerClass * samplesPerSegment + 1;
    testStartIndices = ...
        testFirstIndex + (0:(testSamplesPerClass - 1)) * samplesPerSegment;
    guardSampleCount = testFirstIndex - trainingEndIndex - 1;
    if guardSampleCount < 0
        error('Training and test regions overlap in %s.', sourceFile);
    end

    trainingSample = reshape( ...
        rawSignal(1:trainingEndIndex), ...
        samplesPerSegment, trainingSamplesPerClass);
    testSample = reshape( ...
        rawSignal(testFirstIndex:end), ...
        samplesPerSegment, testSamplesPerClass);

    % Shuffle only the saved column order; raw intervals remain unchanged.
    trainingOrder = randperm(trainingSamplesPerClass);
    testOrder = randperm(testSamplesPerClass);
    trainingSample = trainingSample(:, trainingOrder);
    trainingStartIndices = trainingStartIndices(trainingOrder);
    testSample = testSample(:, testOrder);
    testStartIndices = testStartIndices(testOrder);

    allStartIndices = sort([trainingStartIndices, testStartIndices]);
    if any(diff(allStartIndices) < samplesPerSegment)
        error('Selected one-second samples overlap in %s.', sourceFile);
    end

    SamplingFrequency = samplingFrequency;
    SegmentDurationSeconds = segmentDurationSeconds;
    ClassId = classId;
    ClassName = classNames{classId};

    Sample = trainingSample; %#ok<NASGU>
    StartIndex = trainingStartIndices; %#ok<NASGU>
    EndIndex = StartIndex + samplesPerSegment - 1; %#ok<NASGU>
    trainingPath = fullfile( ...
        trainingDirectory, sprintf('sample%d.mat', classId));
    save(trainingPath, 'Sample', 'StartIndex', 'EndIndex', ...
        'ClassId', 'ClassName', 'sourceFile', 'SamplingFrequency', ...
        'SegmentDurationSeconds', '-v7');

    Sample = testSample; %#ok<NASGU>
    StartIndex = testStartIndices; %#ok<NASGU>
    EndIndex = StartIndex + samplesPerSegment - 1; %#ok<NASGU>
    testPath = fullfile( ...
        testDirectory, sprintf('sample%d.mat', classId));
    save(testPath, 'Sample', 'StartIndex', 'EndIndex', ...
        'ClassId', 'ClassName', 'sourceFile', 'SamplingFrequency', ...
        'SegmentDurationSeconds', '-v7');

    classInfo(classId).classId = classId;
    classInfo(classId).className = ClassName;
    classInfo(classId).sourceFile = sourceFile;
    classInfo(classId).rawSampleCount = rawSampleCount;
    classInfo(classId).trainingSampleCount = trainingSamplesPerClass;
    classInfo(classId).testSampleCount = testSamplesPerClass;
    classInfo(classId).guardSampleCount = guardSampleCount;
    classInfo(classId).guardDurationSeconds = ...
        guardSampleCount / samplingFrequency;
    classInfo(classId).trainingStartIndices = trainingStartIndices;
    classInfo(classId).testStartIndices = testStartIndices;

    fprintf(['%s (sample%d) <- %s: raw=%d, train=%d, test=%d, ' ...
        'guard=%.3f s.\n'], ClassName, classId, sourceFile, ...
        rawSampleCount, trainingSamplesPerClass, testSamplesPerClass, ...
        guardSampleCount / samplingFrequency);
    clear Sample trainingSample testSample rawSignal values header
end

%% Save dataset metadata
metadataPath = fullfile(dataDirectory, 'dataset_metadata.mat');
save(metadataPath, 'classInfo', 'classNames', 'classFileNames', ...
    'samplingFrequency', 'segmentDurationSeconds', 'samplesPerSegment', ...
    'samplesPerClass', 'trainingSamplesPerClass', ...
    'testSamplesPerClass', 'randomSeed', '-v7');

fprintf('\nDataset creation complete.\n');
fprintf('Training pool: %s\n', trainingDirectory);
fprintf('Independent test set: %s\n', testDirectory);
fprintf('Metadata: %s\n', metadataPath);
