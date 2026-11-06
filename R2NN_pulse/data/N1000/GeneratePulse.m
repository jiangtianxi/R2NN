clc;
clear;
close all;

fc = [30 50 70];
Fs = 1e3;
len = 1000;

% Use a different fixed seed for each split. This makes every run
% reproducible while keeping the independent test set separate from the
% random stream used to generate the training set.
splitNames = {'train', 'test', 'experiment'};
splitSizes = [1000, 400, 20];
splitSeeds = [20260810, 20260811, 20260812];

scriptDir = fileparts(mfilename('fullpath'));
t = 0:1/Fs:(len-1)/Fs;
cycle = 11;
NFFT = 2^nextpow2(len);
f = Fs/2*linspace(0, 1, NFFT/2+1);

for splitIdx = 1:numel(splitNames)
    splitName = splitNames{splitIdx};
    num = splitSizes(splitIdx);

    % Reset the generator once per split, before generating any class.
    rng(splitSeeds(splitIdx), 'twister');

    outputDir = fullfile(scriptDir, splitName);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    for classIdx = 1:numel(fc)
        f0 = fc(classIdx);
        delay = cycle/f0;
        cleanSignal = ...
            (sign(t-delay)-sign(t-cycle/f0-delay)) .* ...
            (1-cos(2*pi*f0*t/cycle)) .* sin(2*pi*f0*t);

        Sample = zeros(len, num);
        T = zeros(NFFT, num);

        for sampleIdx = 1:num
            y = awgn(cleanSignal, 0);

            output = y;
            output = (output-mean(output))/max(abs(output)); % Zero-mean, peak-normalized copy (spectrum only)
            Ao = 2*abs(fft(output, NFFT)/len);
            T(:, sampleIdx) = Ao;
            Sample(:, sampleIdx) = y';
        end

        outputFile = fullfile( ...
            outputDir, sprintf('sample%d.mat', classIdx));
        save(outputFile, 'Sample');
        fprintf('Saved %s (%d samples, seed %d).\n', ...
            outputFile, num, splitSeeds(splitIdx));
    end
end

% Quick check: last generated sample and the spectrum of the second sample
% of the last class (uncomment the mean to plot the class-averaged spectrum).
figure(1);
plot(t, y);
tab = [t', y'];

figure(2);
% Tmean = mean(T, 2);
Tmean = T(:,2);
plot(f, Tmean(1:NFFT/2+1), '-', 'LineWidth', 2);
xlim([1, 200]);
title('Frequency response');
xlabel('Frequency (Hz)');
ylabel('Amplitude (a.u.)');
