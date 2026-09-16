% DEMO  Επίδειξη του αλγορίθμου pocket perceptron σε δύο σύνολα δεδομένων.
%
%   Εκτελέστε το από τον ριζικό κατάλογο του repository:
%       >> demo
%
%   Το πρώτο σύνολο είναι γραμμικά διαχωρίσιμο, οπότε ο αλγόριθμος τερματίζει
%   μόλις ταξινομήσει σωστά και τα N πρότυπα. Το δεύτερο δεν είναι, οπότε
%   εξαντλεί τις εποχές και κρατά στην «τσέπη» την καλύτερη λύση που βρήκε —
%   εκεί φαίνεται και ο λόγος ύπαρξης του αλγορίθμου.

clear; close all; clc;

datasets = {'data/separable2D.csv', 'data/nonseparable2D.csv'};

for k = 1:numel(datasets)
    fprintf('==== %s ====\n', datasets{k});

    % Τα αρχεία είναι απλά αριθμητικά CSV (N x 3), οπότε η load() τα διαβάζει
    % τόσο σε MATLAB όσο και σε GNU Octave.
    data = load(datasets{k});

    pocket_perceptron(data, 0.01, 1000, true);
    fprintf('\n');
end
