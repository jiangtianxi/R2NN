% =========================================================================
% Balanced 3-bit sequential parity dataset generator.
%
% Bit-to-frequency mapping:
%   Bit 0 -> 40 Hz
%   Bit 1 -> 60 Hz
%
% The script creates independent training and test splits in a directory
% named after the sequence length, for example:
%   N900/train/sample1.mat  Even-parity training samples
%   N900/train/sample2.mat  Odd-parity training samples
%   N900/test/sample1.mat   Even-parity independent test samples
%   N900/test/sample2.mat   Odd-parity independent test samples
%   N900/dataset_config.mat Complete generation configuration
%
% Each sample MAT file contains:
%   data          - N x num_samples waveform matrix
%   bit_sequences - num_samples x 3 bit matrix
%   class_label   - 0 for even parity or 1 for odd parity
%   metadata      - Split, seed, class, sample count, and pattern counts
% =========================================================================

clear;
clc;
close all;

%% 1. Generation settings
Fs = 1000;
N = 900;
L = 3;
noise_level = 0.15;

num_train_per_class = 1000;
num_test_per_class = 400;

% Different fixed seeds make the splits reproducible and independent.
train_seed = 20260731;
test_seed = 20260801;

% Set this to false when figures are not needed.
make_plot = true;

assert(isscalar(N) && N > 0 && N == floor(N), ...
    'N must be a positive integer.');
assert(L == 3, 'This generator is configured for exactly three bits.');
assert(mod(N, L) == 0, 'N must be divisible by L.');
assert(mod(num_train_per_class, 4) == 0, ...
    ['The training sample count per class must be divisible by four ', ...
     'because each parity class contains four bit patterns.']);
assert(mod(num_test_per_class, 4) == 0, ...
    ['The test sample count per class must be divisible by four ', ...
     'because each parity class contains four bit patterns.']);
assert(train_seed ~= test_seed, ...
    'Training and test splits must use different random seeds.');

script_dir = fileparts(mfilename('fullpath'));
dataset_name = sprintf('N%d', N);
dataset_dir = fullfile(script_dir, dataset_name);
train_dir = fullfile(dataset_dir, 'train');
test_dir = fullfile(dataset_dir, 'test');

if ~exist(dataset_dir, 'dir')
    mkdir(dataset_dir);
end
if ~exist(train_dir, 'dir')
    mkdir(train_dir);
end
if ~exist(test_dir, 'dir')
    mkdir(test_dir);
end

%% 2. Generate the training and independent test splits
fprintf('Generating balanced training data with seed %d...\n', train_seed);
train_set = generate_split( ...
    Fs, N, L, noise_level, num_train_per_class, train_seed, 'train');

fprintf('Generating independent test data with seed %d...\n', test_seed);
test_set = generate_split( ...
    Fs, N, L, noise_level, num_test_per_class, test_seed, 'test');

%% 3. Save the generated data
save_class_file(fullfile(train_dir, 'sample1.mat'), train_set, 0);
save_class_file(fullfile(train_dir, 'sample2.mat'), train_set, 1);
save_class_file(fullfile(test_dir, 'sample1.mat'), test_set, 0);
save_class_file(fullfile(test_dir, 'sample2.mat'), test_set, 1);

config = struct( ...
    'Fs', Fs, ...
    'N', N, ...
    'L', L, ...
    'pts_per_segment', N / L, ...
    'bit_frequencies_hz', [40, 60], ...
    'noise_level', noise_level, ...
    'num_train_per_class', num_train_per_class, ...
    'num_test_per_class', num_test_per_class, ...
    'train_seed', train_seed, ...
    'test_seed', test_seed, ...
    'sample1_class', 'even parity (label 0)', ...
    'sample2_class', 'odd parity (label 1)');
save(fullfile(dataset_dir, 'dataset_config.mat'), 'config', '-v7');

%% 4. Validate dimensions, labels, balance, and split independence
validate_split(train_set, N, num_train_per_class, 'Training split');
validate_split(test_set, N, num_test_per_class, 'Test split');

% Exact overlap should not occur because the splits use different seeds.
% This explicit check protects the independent-test protocol.
all_train = [train_set.class0_data, train_set.class1_data];
all_test = [test_set.class0_data, test_set.class1_data];
has_exact_overlap = any(ismember(all_test', all_train', 'rows'));
assert(~has_exact_overlap, ...
    'An exactly duplicated waveform was found across train and test splits.');

fprintf('\nDataset generation completed successfully.\n');
fprintf('  Training directory: %s (%d samples per class)\n', ...
    train_dir, num_train_per_class);
fprintf('  Test directory: %s (%d samples per class)\n', ...
    test_dir, num_test_per_class);
fprintf('  All four bit patterns in each class have equal counts.\n');
fprintf('  No exactly duplicated waveform exists across the two splits.\n');

%% 5. Optional visualization
if make_plot
    t = (0:N-1) / Fs;
    plot_sample_id = 3;

    assert(plot_sample_id <= num_train_per_class, ...
        'plot_sample_id exceeds the available training sample count.');

    sig_even = train_set.class0_data(:, plot_sample_id);
    sig_odd = train_set.class1_data(:, plot_sample_id);

    % Read the bit labels from the same samples used for the waveforms.
    bits_even_text = sprintf( ...
        '%d%d%d', train_set.class0_bits(plot_sample_id, :));
    bits_odd_text = sprintf( ...
        '%d%d%d', train_set.class1_bits(plot_sample_id, :));

    figure( ...
        'Name', 'Balanced 3-Bit Sequential Parity Dataset', ...
        'Position', [100, 100, 1100, 600]);

    subplot(2, 2, 1);
    plot(t, sig_even, 'b', 'LineWidth', 1);
    add_segment_lines(L, N / Fs);
    title(sprintf('Parity = 0, bits = %s', bits_even_text));
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, N / Fs]);
    grid on;

    subplot(2, 2, 3);
    spectrogram(sig_even, kaiser(128, 5), 120, 256, Fs, 'yaxis');
    title('Parity = 0, short-time spectrum');
    ylim([0, 120]);

    subplot(2, 2, 2);
    plot(t, sig_odd, 'k', 'LineWidth', 1);
    add_segment_lines(L, N / Fs);
    title(sprintf('Parity = 1, bits = %s', bits_odd_text));
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([0, N / Fs]);
    grid on;

    subplot(2, 2, 4);
    spectrogram(sig_odd, kaiser(128, 5), 120, 256, Fs, 'yaxis');
    title('Parity = 1, short-time spectrum');
    ylim([0, 120]);

    sgtitle('Balanced 3-Bit Sequential Parity Dataset');
end

%% Local functions
function split = generate_split( ...
        Fs, N, L, noise_level, num_per_class, seed, split_name)
    % Generate one balanced split from a deterministic random seed.
    rng(seed, 'twister');

    pts_per_segment = N / L;
    all_patterns = dec2bin(0:(2^L - 1), L) - '0';
    pattern_parity = mod(sum(all_patterns, 2), 2);
    patterns_even = all_patterns(pattern_parity == 0, :);
    patterns_odd = all_patterns(pattern_parity == 1, :);

    [class0_data, class0_bits] = generate_class( ...
        patterns_even, num_per_class, Fs, N, ...
        pts_per_segment, noise_level);
    [class1_data, class1_bits] = generate_class( ...
        patterns_odd, num_per_class, Fs, N, ...
        pts_per_segment, noise_level);

    split = struct( ...
        'name', split_name, ...
        'seed', seed, ...
        'class0_data', class0_data, ...
        'class0_bits', class0_bits, ...
        'class1_data', class1_data, ...
        'class1_bits', class1_bits);
end

function [data, bit_sequences] = generate_class( ...
        class_patterns, num_samples, Fs, N, pts_per_segment, noise_level)
    % Repeat every pattern equally, then shuffle the complete balanced list.
    num_patterns = size(class_patterns, 1);
    repeats_per_pattern = num_samples / num_patterns;

    bit_sequences = repelem(class_patterns, repeats_per_pattern, 1);
    bit_sequences = bit_sequences(randperm(num_samples), :);
    data = zeros(N, num_samples);

    for sample_id = 1:num_samples
        bits = bit_sequences(sample_id, :);
        frequencies = 40 + 20 * bits;
        frequency_vector = repelem(frequencies, pts_per_segment);

        % Preserve phase continuity across all three bit segments.
        initial_phase = 2 * pi * rand();
        phase = initial_phase + 2 * pi * cumsum(frequency_vector) / Fs;
        signal = sin(phase) + noise_level * randn(1, N);
        data(:, sample_id) = signal';
    end
end

function save_class_file(file_path, split, class_label)
    % Save one parity class while preserving waveform-to-bit correspondence.
    if class_label == 0
        data = split.class0_data;
        bit_sequences = split.class0_bits;
    else
        data = split.class1_data;
        bit_sequences = split.class1_bits;
    end

    pattern_ids = bit_sequences * [4; 2; 1];
    pattern_counts = accumarray(pattern_ids + 1, 1, [8, 1])';
    metadata = struct( ...
        'split', split.name, ...
        'rng_algorithm', 'twister', ...
        'rng_seed', split.seed, ...
        'class_label', class_label, ...
        'num_samples', size(data, 2), ...
        'pattern_counts_for_000_to_111', pattern_counts);

    save(file_path, ...
        'data', 'bit_sequences', 'class_label', 'metadata', '-v7');
    fprintf('  Saved %s: data shape = %d x %d\n', ...
        file_path, size(data, 1), size(data, 2));
end

function validate_split( ...
        split, expected_length, expected_per_class, split_description)
    % Validate array sizes, parity labels, and per-pattern balance.
    assert(isequal( ...
        size(split.class0_data), ...
        [expected_length, expected_per_class]), ...
        'Class 0 waveform dimensions are incorrect.');
    assert(isequal( ...
        size(split.class1_data), ...
        [expected_length, expected_per_class]), ...
        'Class 1 waveform dimensions are incorrect.');
    assert(isequal( ...
        size(split.class0_bits), [expected_per_class, 3]), ...
        'Class 0 bit-sequence dimensions are incorrect.');
    assert(isequal( ...
        size(split.class1_bits), [expected_per_class, 3]), ...
        'Class 1 bit-sequence dimensions are incorrect.');

    assert(all(mod(sum(split.class0_bits, 2), 2) == 0), ...
        'Class 0 contains a bit sequence with odd parity.');
    assert(all(mod(sum(split.class1_bits, 2), 2) == 1), ...
        'Class 1 contains a bit sequence with even parity.');

    even_counts = count_patterns(split.class0_bits);
    odd_counts = count_patterns(split.class1_bits);
    expected_per_pattern = expected_per_class / 4;

    assert(all(even_counts(even_counts > 0) == expected_per_pattern), ...
        'Even-parity bit patterns are not equally represented.');
    assert(all(odd_counts(odd_counts > 0) == expected_per_pattern), ...
        'Odd-parity bit patterns are not equally represented.');

    fprintf( ...
        ['  %s validated: class 0 = %d, class 1 = %d, ', ...
         'samples per pattern = %d\n'], ...
        split_description, size(split.class0_data, 2), ...
        size(split.class1_data, 2), expected_per_pattern);
end

function counts = count_patterns(bit_sequences)
    % Count binary patterns in the order 000, 001, ..., 111.
    pattern_ids = bit_sequences * [4; 2; 1];
    counts = accumarray(pattern_ids + 1, 1, [8, 1]);
end

function add_segment_lines(segment_count, total_duration)
    % Mark the boundaries between consecutive bit segments.
    for segment_id = 1:(segment_count - 1)
        xline( ...
            segment_id * total_duration / segment_count, ...
            'r--', ...
            'LineWidth', 1.2);
    end
end
