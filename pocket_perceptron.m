function [best_weights, best_correct] = pocket_perceptron(input, learning_rate, max_epochs, do_plot)
%POCKET_PERCEPTRON Αλγόριθμος pocket perceptron (Gallant, 1990).
%
%   [W, C] = POCKET_PERCEPTRON(INPUT) εκπαιδεύει ένα perceptron ενός επιπέδου
%   με τον αλγόριθμο pocket. Το INPUT είναι πίνακας N x 3 με στήλες
%   [x1 x2 target], όπου target ∈ {-1, +1}. Επιστρέφει το διάνυσμα βαρών της
%   «τσέπης» W = [w1; w2; bias] και το πλήθος C των προτύπων που το W
%   ταξινομεί σωστά.
%
%   [W, C] = POCKET_PERCEPTRON(INPUT, LEARNING_RATE, MAX_EPOCHS, DO_PLOT)
%   ορίζει τον ρυθμό εκμάθησης (προεπιλογή 0.01), το μέγιστο πλήθος εποχών
%   (προεπιλογή 1000) και αν θα γίνει σχεδίαση (προεπιλογή true).
%
%   Συνθήκη τερματισμού
%   -------------------
%   Ο αλγόριθμος perceptron δεν τερματίζει από μόνος του όταν οι κλάσεις δεν
%   είναι γραμμικά διαχωρίσιμες. Εδώ η επανάληψη σταματά μόλις το διάνυσμα
%   της τσέπης ταξινομήσει σωστά και τα N πρότυπα (C == N), δηλαδή μόλις
%   βρεθεί πλήρης γραμμικός διαχωρισμός. Αν τέτοιος διαχωρισμός δεν υπάρχει,
%   η επανάληψη φράσσεται από το MAX_EPOCHS και η τσέπη κρατά την καλύτερη
%   λύση που συναντήθηκε.
%
%   Παράδειγμα:
%       data = load('data/nonseparable2D.csv');
%       [w, c] = pocket_perceptron(data);
%
%   Δείτε επίσης: DEMO.

    % ---- Προεπιλεγμένα ορίσματα ------------------------------------------
    if nargin < 2 || isempty(learning_rate), learning_rate = 0.01; end
    if nargin < 3 || isempty(max_epochs),    max_epochs    = 1000; end
    if nargin < 4 || isempty(do_plot),       do_plot       = true; end

    % ---- Δεδομένα εισόδου -------------------------------------------------
    if size(input, 2) ~= 3
        error('pocket_perceptron:badInput', ...
              'Το input πρέπει να είναι πίνακας N x 3 με στήλες [x1 x2 target].');
    end

    x1     = input(:, 1);            % Χαρακτηριστικό x1
    x2     = input(:, 2);            % Χαρακτηριστικό x2
    target = input(:, 3);            % Στόχοι (-1 ή 1)

    if any(target ~= 1 & target ~= -1)
        error('pocket_perceptron:badTargets', ...
              'Οι στόχοι (3η στήλη) πρέπει να είναι -1 ή 1.');
    end

    num_samples = size(input, 1);    % Αριθμός δειγμάτων

    % Επαυξημένος πίνακας προτύπων: κάθε γραμμή είναι [x1 x2 1], ώστε το bias
    % να αντιμετωπίζεται σαν ένα ακόμη βάρος (το τρίτο).
    X = [x1, x2, ones(num_samples, 1)];

    % ---- Αρχικοποίηση -----------------------------------------------------
    weights      = [0; 0; 0];        % Αρχικοποίηση βαρών σε 0
    best_weights = weights;          % Βάρη της «τσέπης» (pocket weights)
    best_correct = count_correct(X, target, weights);   % και το πλήθος τους

    % ---- Σχεδίαση: δεδομένα -----------------------------------------------
    pad   = 1;
    x_lim = [min(x1) - pad, max(x1) + pad];
    y_lim = [min(x2) - pad, max(x2) + pad];

    if do_plot
        figure('Name', 'Pocket Perceptron');
        hold on;
        grid on;
        class_minus = scatter(x1(target == -1), x2(target == -1), 120, 'r', 'x', 'LineWidth', 2);
        class_plus  = scatter(x1(target ==  1), x2(target ==  1), 120, 'b', 'o', 'LineWidth', 2);
        axis([x_lim, y_lim]);
        xlabel('x_1');
        ylabel('x_2');
        title('Pocket Perceptron');
    end
    current_line = [];

    % ---- Εκπαίδευση -------------------------------------------------------
    for epoch = 1:max_epochs
        for i = 1:num_samples
            % Πρόβλεψη με τα τρέχοντα βάρη
            prediction = classify(X(i, :) * weights);
            error_i    = target(i) - prediction;

            if error_i ~= 0
                % Ενημέρωση βαρών (κανόνας perceptron) μόνο σε λάθος πρόβλεψη
                weights = weights + learning_rate * error_i * X(i, :)';

                % --- Το βήμα της «τσέπης» ---
                % Τα ΝΕΑ βάρη αξιολογούνται σε ΟΛΟ το σύνολο εκπαίδευσης και
                % μπαίνουν στην τσέπη μόνο αν ταξινομούν σωστά περισσότερα
                % πρότυπα από ό,τι τα βάρη που βρίσκονται ήδη εκεί.
                correct = count_correct(X, target, weights);
                if correct > best_correct
                    best_correct = correct;
                    best_weights = weights;
                end
            end
        end

        % Η διαχωριστική ευθεία της τρέχουσας επανάληψης· η προηγούμενη
        % ξεθωριάζει και μένει σαν ίχνος της πορείας του αλγορίθμου.
        if do_plot
            current_line = fade_and_draw(current_line, weights, x_lim, y_lim);
            drawnow;
        end

        % Συνθήκη τερματισμού: πλήρης γραμμικός διαχωρισμός
        if best_correct == num_samples
            break;
        end
    end

    % ---- Σχεδίαση: τελική διαχωριστική ευθεία -----------------------------
    if do_plot
        fade_and_draw(current_line, [], x_lim, y_lim);
        final_line = plot_boundary(best_weights, x_lim, y_lim, 'r-', 'LineWidth', 2);

        handles = [class_minus, class_plus];
        labels  = {'Class -1', 'Class +1'};
        if ~isempty(final_line)
            handles(end + 1) = final_line;
            labels{end + 1}  = 'Pocket decision boundary';
        end
        legend(handles, labels, 'Location', 'best');
    end

    % ---- Αποτελέσματα -----------------------------------------------------
    fprintf('Εποχές που εκτελέστηκαν: %d\n', epoch);
    fprintf('Βάρη της τσέπης: w1 = %.4f, w2 = %.4f, bias = %.4f\n', ...
            best_weights(1), best_weights(2), best_weights(3));
    fprintf('Σωστά ταξινομημένα πρότυπα: %d/%d (%.1f%%)\n', ...
            best_correct, num_samples, 100 * best_correct / num_samples);
    if best_correct == num_samples
        fprintf('Τερματισμός: όλα τα πρότυπα ταξινομήθηκαν σωστά.\n');
    else
        fprintf(['Τερματισμός: συμπληρώθηκαν %d εποχές χωρίς πλήρη διαχωρισμό ' ...
                 '(οι κλάσεις δεν φαίνονται γραμμικά διαχωρίσιμες).\n'], max_epochs);
    end
end

% =========================================================================
% Βοηθητικές συναρτήσεις
% =========================================================================

function labels = classify(weighted_sum)
%CLASSIFY Συνάρτηση ενεργοποίησης (step): επιστρέφει -1 ή +1.
%   Το sign() επιστρέφει 0 όταν το άθροισμα είναι ακριβώς 0· με σύμβαση το
%   αντιμετωπίζουμε ως -1, ώστε η έξοδος να είναι πάντα δυαδική.
    labels = sign(weighted_sum);
    labels(labels == 0) = -1;
end

function n = count_correct(X, target, weights)
%COUNT_CORRECT Πλήθος προτύπων που ταξινομεί σωστά ένα διάνυσμα βαρών.
    n = sum(classify(X * weights) == target);
end

function h = plot_boundary(weights, x_lim, y_lim, varargin)
%PLOT_BOUNDARY Σχεδιάζει την ευθεία w1*x1 + w2*x2 + bias = 0.
    if weights(2) ~= 0
        xs = x_lim;
        ys = -(weights(1) * xs + weights(3)) / weights(2);
    elseif weights(1) ~= 0
        % Κατακόρυφη ευθεία: x1 = -bias / w1
        xs = [-weights(3) / weights(1), -weights(3) / weights(1)];
        ys = y_lim;
    else
        % Μηδενικά βάρη: δεν ορίζεται ευθεία.
        h = [];
        return;
    end
    h = plot(xs, ys, varargin{:});
end

function h = fade_and_draw(previous_line, weights, x_lim, y_lim)
%FADE_AND_DRAW Ξεθωριάζει την προηγούμενη ευθεία και σχεδιάζει τη νέα.
    if ~isempty(previous_line) && ishandle(previous_line)
        set(previous_line, 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5);
    end
    if isempty(weights)
        h = [];
    else
        h = plot_boundary(weights, x_lim, y_lim, 'k--', 'LineWidth', 1);
    end
end
